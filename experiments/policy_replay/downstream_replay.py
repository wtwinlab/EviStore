"""Counterfactual re-aggregation of the sealed per-round deltas (test accuracy).

The deltas of each round are computed against the deployed model state, so
re-aggregating them under a different exclusion mask is exact only while the
counterfactual trajectory stays on the deployed one. Model-level consequences of
a revision are evaluated by certified federated repair (dtrust/fl/cfr.py), which
re-aggregates or retrains from a verified checkpoint.

This script reports a number outside the sealed log: the **test accuracy** of the
global model under each exclusion policy. It runs the DEPLOYED policy's FL
trajectory once (clients train on the v1-aggregated model), captures each round's
realized per-client deltas, and COUNTERFACTUALLY re-aggregates the SAME deltas
under four exclusion masks:
  none   : FedAvg over all clients (no defence, lower bound)
  v1     : exclude SPRT-banned under rho_v1 (the deployed/lenient policy)
  v2     : exclude SPRT-banned under rho_v2 (the tightened audit policy)
  oracle : exclude the ground-truth malicious set (upper bound)
and reports each model's test accuracy over the incident. Delta_acc(v2 - v1) is
the downstream value of the policy revision.

Online trust: lambda_t = decay*lambda_{t-1} + sum LLR (shared across policies;
v1/v2 differ only in the lambda_min threshold), exactly as sprt_replay.replay.
Evidence mapping from (cos, norm) matches real_event (measured-client: att blind,
grad <- (cos+1)/2, perf <- cos>=median, beh <- norm spike, seal=1).

Run on the GPU server (needs torch + data):
    python -m experiment_scripts.dtrust_experiments.c1_audit.downstream_replay \
        --seeds 0 1 2 3 4 --lf_flip_frac 0.03 --decay 0.95 --h_v1 15 \
        --out results/recovery/c1_downstream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from dtrust.ml.models.registry import create_model
from dtrust.ml.utils.data_utils import load_fashion_mnist_data
from scripts.run_recovery_experiment import (
    set_seed, iid_partition, ClientPool, _train_round_deltas,
    compute_cos_to_median, apply_delta, evaluate, flat_params, set_flat_params,
)
from dtrust.trust._llr import llr_att, llr_grad, llr_perf, llr_beh, llr_seal

from .policies import policy_pair, load_base_config


def _round_llr_increment(cfg, ch, clamp, rng, cos_row, norm_row):
    """Per-client raw LLR increment for one round (measured-client mapping)."""
    N = len(cos_row)
    ref = float(np.median(cos_row))
    nmean, nstd = float(np.mean(norm_row)), max(float(np.std(norm_row)), 1e-6)
    inc = np.zeros(N)
    for i in range(N):
        g = min(max((cos_row[i] + 1.0) / 2.0, 1e-3), 1 - 1e-3)
        att = rng.gauss(ch["att"]["mu0"], ch["att"]["sigma0"])     # blind
        perf = 1 if cos_row[i] >= ref else 0
        beh = "susp" if norm_row[i] > nmean + 3.0 * nstd else "none"
        l_a = max(-clamp, min(clamp, llr_att(att, ch["att"]["mu0"], ch["att"]["sigma0"], ch["att"]["mu1"])))
        l_g = max(-clamp, min(clamp, llr_grad(g, ch["grad"]["alpha0"], ch["grad"]["beta0"], ch["grad"]["alpha1"], ch["grad"]["beta1"])))
        l_p = max(-clamp, min(clamp, llr_perf(perf, ch["perf"]["p0"], ch["perf"]["p1"])))
        l_b = max(-clamp, min(clamp, llr_beh(beh, ch["beh"])))
        l_s = max(-clamp * 10.0, min(clamp, llr_seal(1, ch["seal"])))
        inc[i] = l_a + l_g + l_p + l_b + l_s
    return inc


def _agg(deltas, mask):
    keep = [deltas[i] for i in range(len(deltas)) if mask[i]]
    if not keep:
        return torch.zeros_like(deltas[0])
    return torch.stack(keep).mean(dim=0)


def run_seed(seed, theta_v1, theta_v2, malicious_frac, lf_flip_frac, r_atk, r_inc,
             n_clients, device):
    cfg = SimpleNamespace(
        n_clients=n_clients, attack="lf", lf_flip_frac=lf_flip_frac,
        attack_ramp=False, attack_intensity=1.0, r_atk=r_atk, r_inc=r_inc,
        num_classes=10, local_steps=5, lr=0.05, model_name="simple_nn_mnist")
    set_seed(seed)
    Xtr, ytr, Xte, yte = load_fashion_mnist_data()
    parts = iid_partition(len(Xtr), n_clients, seed)
    pool = ClientPool(Xtr, ytr, parts, 64, device)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(Xte).float(),
                                           torch.from_numpy(yte).long()),
                             batch_size=512, shuffle=False)
    rng_np = np.random.default_rng(seed)
    n_mal = int(round(malicious_frac * n_clients))
    malicious = set(int(x) for x in rng_np.choice(n_clients, n_mal, replace=False))

    # Four counterfactual models, all fed the SAME realized deltas.
    models = {k: create_model(cfg.model_name).to(device) for k in ("none", "v1", "v2", "oracle")}
    init = flat_params(models["none"]).clone()
    for m in models.values():
        set_flat_params(m, init)
    deployed = models["v1"]          # clients train on the deployed (v1) model
    mtmp = create_model(cfg.model_name).to(device)

    ch = theta_v1["channels"]; clamp = theta_v1["thresholds"]["clamp_per_llr"]
    clamp_lam = theta_v1["thresholds"]["clamp_lambda"]; decay = float(theta_v1["decay"])
    lmin_v1 = theta_v1["thresholds"]["lambda_min"]; lmin_v2 = theta_v2["thresholds"]["lambda_min"]
    susp_max = theta_v1["quarantine"]["susp_max"]
    import random as _r
    trng = _r.Random(10_000 + seed)

    lam = np.zeros(n_clients)
    streak = {"v1": np.zeros(n_clients, int), "v2": np.zeros(n_clients, int)}
    banned = {"v1": np.zeros(n_clients, bool), "v2": np.zeros(n_clients, bool)}
    curves = {k: [] for k in models}

    import torch as _t
    for rnd in range(r_inc + 1):
        set_flat_params(mtmp, flat_params(deployed))
        # train all clients on the deployed model
        deltas, _benign, _prov = _train_round_deltas(cfg, deployed, mtmp, pool, malicious, rnd, device)
        cos, norms = compute_cos_to_median(deltas)
        inc = _round_llr_increment(cfg, ch, clamp, trng, cos, norms)
        lam = np.clip(decay * lam + inc, -clamp_lam, clamp_lam)
        for pol, lmin in (("v1", lmin_v1), ("v2", lmin_v2)):
            sub = lam <= lmin
            streak[pol] = np.where(sub, streak[pol] + 1, 0)
            banned[pol] = banned[pol] | (streak[pol] >= susp_max)
        masks = {
            "none": [True] * n_clients,
            "v1": [not banned["v1"][i] for i in range(n_clients)],
            "v2": [not banned["v2"][i] for i in range(n_clients)],
            "oracle": [i not in malicious for i in range(n_clients)],
        }
        for k, m in models.items():
            apply_delta(m, _agg(deltas, masks[k]))
        if rnd % 10 == 0 or rnd == r_inc:
            for k, m in models.items():
                curves[k].append((rnd, evaluate(m, test_loader, device)))

    final = {k: evaluate(m, test_loader, device) for k, m in models.items()}
    return {"seed": seed, "malicious": sorted(malicious),
            "final_acc": final, "curves": curves,
            "banned_v1": int(banned["v1"].sum()), "banned_v2": int(banned["v2"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lf_flip_frac", type=float, default=0.03)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--h_v1", type=float, default=15.0)
    ap.add_argument("--malicious_frac", type=float, default=0.2)
    ap.add_argument("--r_atk", type=int, default=50)
    ap.add_argument("--r_inc", type=int, default=150)
    ap.add_argument("--n_clients", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default="results/recovery/c1_downstream")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    theta_v1, theta_v2 = policy_pair(load_base_config(), h_v1=args.h_v1, decay=args.decay)
    results = [run_seed(s, theta_v1, theta_v2, args.malicious_frac, args.lf_flip_frac,
                        args.r_atk, args.r_inc, args.n_clients, device) for s in args.seeds]

    keys = ("none", "v1", "v2", "oracle")
    agg = {k: (float(np.mean([r["final_acc"][k] for r in results])),
               float(np.std([r["final_acc"][k] for r in results]))) for k in keys}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": results, "final_acc_mean_std": agg,
               "delta_acc_v2_minus_v1": agg["v2"][0] - agg["v1"][0]},
              open(out / "downstream.json", "w"), indent=2)
    print("\n==== T1-b DOWNSTREAM TEST ACCURACY (final, mean±std over seeds) ====")
    for k in keys:
        print(f"  {k:7}: {agg[k][0]:.4f} ± {agg[k][1]:.4f}")
    print(f"  --> Δacc(v2 − v1) = {agg['v2'][0]-agg['v1'][0]:+.4f}   "
          f"(v2 closes {100*(agg['v2'][0]-agg['v1'][0])/max(agg['oracle'][0]-agg['v1'][0],1e-6):.0f}% of the v1→oracle gap)")
    print(f"  wrote {out/'downstream.json'}")


if __name__ == "__main__":
    main()
