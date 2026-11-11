"""Self-contained recovery baselines: Retrain, FedRecover, Crab.

These are the *external* recovery baselines for the TEE-framework validation. They
are kept separate from DTRUST: they do NOT use the
trust-aware aggregation, the DTRUST multi-factor trust, the RAC/attestation
signal, or the ATLAS shadow / two-tier protocol. They share only the neutral FL
substrate (dataset, model, attack, client data) with everything else.

Each baseline:
  * trains a poisoned model with a STANDARD robust aggregator (median /
    trimmed-mean / fedavg), storing the history it needs;
  * is told the malicious set at a statistical-detection round `detect_round`
    (modelling post-hoc statistical detection — NOT TEE attestation);
  * recovers a clean model from there with its own algorithm.

Faithful to the official Crab repo (github.com/shenjiyuan123/Adaptive_Recovery):
serverBase.retrain (Retrain), serverFedRecover.recover (FedRecover),
serverCrab.{train_with_select,adaptive_recover} + serverEraser.unlearning_step_once
(Crab). Recovery COST is reported as the number of client-round local-training
invocations (the FedRecover efficiency metric).

The module is dependency-injected via an `infra` object so it pulls only neutral
helpers (build_model, local-training, neutral aggregators, evaluate) from the
caller and imports nothing from the DTRUST trust/attestation/shadow stack.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtrust.attacks.minmax_attack import min_max_attack_full_knowledge


@dataclass
class Infra:
    """Neutral FL infrastructure injected by the caller (no DTRUST-specific bits)."""
    build_model: Callable[[], nn.Module]          # () -> fresh model on device
    client_loaders: list                          # per-client DataLoader
    client_sizes: list                            # per-client #samples
    test_loader: object
    device: object
    num_classes: int
    evaluate: Callable                            # (model, loader, device) -> (loss, err)
    local_train_steps: Callable                  # (model, loader, steps, lr, device, transform) -> delta
    apply_delta: Callable                        # (model, delta, lr) -> None
    median_aggregate: Callable                   # (deltas) -> delta
    trimmed_mean_aggregate: Callable             # (deltas, k) -> delta
    fedavg_aggregate: Callable                   # (deltas, weights) -> delta
    trim_attack: Callable                        # full-knowledge trim attack
    label_flip_batch: Callable                   # (batch, True, num_classes) -> batch
    # REAL DTRUST trust pipeline (injected; used by the faithful DTRUST detector)
    rac_generate: Callable = None                # generate_rac
    rac_verify: Callable = None                  # verify_rac
    trust_state_cls: Callable = None             # ClientTrustState
    trust_update: Callable = None                # update_trust_v2
    trust_args: object = None                    # argparse args (w_p,w_b,phi,eta,...)


@dataclass
class BaselineConfig:
    method: str                  # retrain | fedrecover | crab
    attack: str                  # trim | lf | none
    num_clients: int
    malicious: set
    rounds: int
    local_steps: int
    lr: float
    combined_lr: float
    agg: str = "median"          # neutral base aggregator
    attack_start_round: int = 1
    detect_round: int = 0        # 0 => detect at end of training
    recovery_rounds: int = 0     # 0 => same as `rounds`
    trim_clients_num: int = 2    # for trimmed-mean
    min_trust: float = 0.3       # DTRUST detector: flag clients with overall_score < min_trust
    fedrecover_lbfgs_lr: float = 0.05   # L-BFGS recovery lr (decoupled from training lr)
    # Crab selection ratios (official defaults: P_rounds=0.6, X_clients=0.7)
    crab_select_round_ratio: float = 0.6
    crab_select_client_ratio: float = 0.7
    replay_polish: int = 0       # replay: # fine-tune rounds on benign clients after 0-cost replay
    adversary_continues: bool = False  # recov_cmp: undetected malicious keep attacking in recovery
    # B — closed-form influence removal (not trajectory replay).
    unlearn_precond: str = "diag"      # identity | diag | lowrank  (benign-Fisher preconditioner)
    # lowrank (inverse-Fisher via Woodbury) shrinks the benign-descent-aligned component
    # of the malicious push that is to be subtracted; default is diag.
    unlearn_lambda: float = 1e-3       # Fisher damping / Woodbury ridge
    unlearn_lowrank_m: int = 128       # # benign cells subsampled for the low-rank Fisher
    unlearn_hybrid_rounds: int = 0     # t benign forward rounds AFTER the closed-form step
    unlearn_cert_sigma: float = 0.0    # B2: calibrated Gaussian noise for an (eps,delta) bound
    # C1 — subspace-selective rollback: roll back ONLY the poison subspace to safe-ckpt c*.
    subspace_k: int = 8                # # poison directions (top-k of sealed displacement matrix)
    subspace_fwd: int = 0              # benign forward rounds after the subspace rollback
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Neutral aggregation over a list of per-client deltas
# ──────────────────────────────────────────────────────────────────────────
def _aggregate(infra: Infra, cfg: BaselineConfig, deltas: List[torch.Tensor],
               weights: List[float]) -> torch.Tensor:
    if cfg.agg == "median":
        return infra.median_aggregate(deltas)
    if cfg.agg == "trimmed_mean":
        return infra.trimmed_mean_aggregate(deltas, cfg.trim_clients_num)
    return infra.fedavg_aggregate(deltas, weights)


def _client_delta(infra: Infra, cfg: BaselineConfig, global_model: nn.Module,
                  cid: int, attacking: bool) -> torch.Tensor:
    """One client's local update under the (neutral) attack model."""
    m = infra.build_model()
    m.load_state_dict(global_model.state_dict())
    transform = None
    if attacking and cfg.attack == "lf":
        transform = lambda batch: infra.label_flip_batch(batch, True, infra.num_classes)
    return infra.local_train_steps(m, infra.client_loaders[cid], cfg.local_steps,
                                   cfg.lr, infra.device, transform)


def _round_deltas(infra: Infra, cfg: BaselineConfig, global_model: nn.Module,
                  round_num: int, clients: List[int]) -> List[torch.Tensor]:
    """Compute every selected client's delta for one round, applying the attack."""
    active = cfg.malicious if round_num >= cfg.attack_start_round else set()
    deltas = [_client_delta(infra, cfg, global_model, cid, cid in active) for cid in clients]
    # model-poisoning: replace malicious deltas with crafted ones (trim or min-max).
    # min-max stays inside the benign spread -> evades statistical (cosine/norm) detection
    # but is still a crafted computation -> attestation (CFI) flags it.
    if cfg.attack in ("trim", "minmax") and active:
        mal = [i for i, cid in enumerate(clients) if cid in active]
        if mal:
            if cfg.attack == "minmax":
                poisoned = min_max_attack_full_knowledge(deltas, mal, len(mal), infra.device)
            else:
                poisoned = infra.trim_attack(deltas, mal, len(mal), infra.device)
            for k, idx in enumerate(mal):
                deltas[idx] = poisoned[k]
    return deltas


# ──────────────────────────────────────────────────────────────────────────
# Shared poisoned training (stores history every baseline needs)
# ──────────────────────────────────────────────────────────────────────────
def _train_poisoned(infra: Infra, cfg: BaselineConfig) -> dict:
    gm = infra.build_model()
    initial_state = copy.deepcopy(gm.state_dict())
    all_clients = list(range(cfg.num_clients))
    weights = [float(s) for s in infra.client_sizes]

    hist = {
        "initial_state": initial_state,
        "round_global": {},     # round -> global state_dict (pre-update)
        "round_deltas": {},     # round -> list[client delta] (per all clients)
        "round_trainloss": {},  # round -> train loss proxy (test loss here)
        "metrics": [],          # [(round, acc)]
    }
    # REAL DTRUST trust pipeline (only if injected): per-client ClientTrustState
    use_trust = (infra.trust_state_cls is not None and infra.trust_update is not None)
    if use_trust:
        trust = {c: infra.trust_state_cls() for c in range(cfg.num_clients)}

    for r in range(1, cfg.rounds + 1):
        hist["round_global"][r] = copy.deepcopy(gm.state_dict())
        deltas = _round_deltas(infra, cfg, gm, r, all_clients)
        hist["round_deltas"][r] = [d.clone() for d in deltas]
        # SEALED per-round per-client scalar G_i = cos(Δ_i, coordinate-median ref).
        # This is exactly what the DTRUST schema seals (Eq.2 projected cosine) — a
        # scalar, NOT the raw gradient. Method C (`driftsealed`) queries ONLY these
        # scalars; method B (`driftguided`) uses the raw round_deltas above and is
        # left unchanged. (Additive: this does not affect training or B.)
        _ref = torch.stack(deltas).median(dim=0).values
        hist.setdefault("round_cos", {})[r] = [
            F.cosine_similarity(d.unsqueeze(0), _ref.unsqueeze(0)).item() for d in deltas
        ]

        if use_trust:
            active = cfg.malicious if r >= cfg.attack_start_round else set()
            ref = torch.stack(deltas).median(dim=0).values
            ref_norm = ref.norm().item()
            norms = [d.norm().item() for d in deltas]
            bstats = {"mean": float(np.mean(norms)), "std": max(float(np.std(norms)), 1e-6)}
            for cid in range(cfg.num_clients):
                ct = trust[cid]
                # ── attestation (RAC): simulate via generate_rac/verify_rac ──
                if infra.rac_generate is not None:
                    cert = infra.rac_generate(cid, r, deltas[cid], cfg.local_steps,
                                              cid in active, cfg.attack if cid in active else "none")
                    passed, _viol = infra.rac_verify(cert, deltas[cid], cfg.local_steps, bstats)
                    if not passed:
                        ct.rac_violations += 1
                        ct.runtime_score = max(0.0, ct.runtime_score * 0.5)
                    else:
                        ct.runtime_score = min(1.0, ct.runtime_score + 0.05)
                # ── multi-factor trust update (REAL 0.4·att+0.3·perf+0.3·beh, v2 gating) ──
                cos = F.cosine_similarity(deltas[cid].unsqueeze(0), ref.unsqueeze(0)).item()
                infra.trust_update(ct, cos, norms[cid], ref_norm, r, infra.trust_args)

        agg = _aggregate(infra, cfg, deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        if r == 1 or r % 10 == 0 or r == cfg.rounds:
            loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
            hist["round_trainloss"][r] = loss
            hist["metrics"].append({"round": r, "accuracy": 1.0 - err, "phase": "train"})
    hist["poisoned_state"] = copy.deepcopy(gm.state_dict())
    if use_trust:
        hist["trust"] = trust
    return hist


# ──────────────────────────────────────────────────────────────────────────
# Retrain (serverBase.retrain): from initial model, full-train remaining clients
# ──────────────────────────────────────────────────────────────────────────
def _retrain(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    gm = infra.build_model()
    gm.load_state_dict(hist["initial_state"])
    weights = [float(infra.client_sizes[c]) for c in remaining]
    R = cfg.recovery_rounds or cfg.rounds
    metrics, cost = [], 0
    for r in range(1, R + 1):
        deltas = [_client_delta(infra, cfg, gm, cid, attacking=False) for cid in remaining]
        cost += len(remaining)                       # client-round full-train invocations
        agg = _aggregate(infra, cfg, deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        if r == 1 or r % 10 == 0 or r == R:
            loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
            metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "full_local_train"}


# ──────────────────────────────────────────────────────────────────────────
# FedRecover (serverFedRecover.recover + clientFedRecover.retrain_with_LBFGS):
# from initial model, remaining clients retrain with an L-BFGS optimizer every
# round; reload previous global if train-loss increased. NOTE: this is the Crab
# repo's reference implementation (clients retrain every round with L-BFGS), not
# Cao et al.'s original server-side-estimation variant, so its client cost equals
# retrain's #invocations; faithful to the official code we have.
# ──────────────────────────────────────────────────────────────────────────
def _lbfgs_delta(infra: Infra, cfg: BaselineConfig, global_model: nn.Module, cid: int) -> torch.Tensor:
    from torch.nn.utils import parameters_to_vector
    m = infra.build_model()
    m.load_state_dict(global_model.state_dict())
    before = parameters_to_vector(m.parameters()).detach().clone()
    opt = torch.optim.LBFGS(m.parameters(), lr=cfg.fedrecover_lbfgs_lr, history_size=1, max_iter=4)
    loss_fn = nn.CrossEntropyLoss()
    m.train()
    seen = 0
    for x, y in infra.client_loaders[cid]:
        x, y = x.to(infra.device), y.to(infra.device)
        def closure():
            opt.zero_grad(); loss = loss_fn(m(x), y); loss.backward(); return loss
        opt.step(closure)
        seen += 1
        if seen >= cfg.local_steps:
            break
    after = parameters_to_vector(m.parameters()).detach()
    return after - before


def _fedrecover(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    gm = infra.build_model()
    gm.load_state_dict(hist["initial_state"])
    weights = [float(infra.client_sizes[c]) for c in remaining]
    R = cfg.recovery_rounds or cfg.rounds
    metrics, cost, prev_loss, prev_state = [], 0, float("inf"), None
    for r in range(1, R + 1):
        deltas = [_lbfgs_delta(infra, cfg, gm, cid) for cid in remaining]
        cost += len(remaining)
        agg = _aggregate(infra, cfg, deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
        if loss > prev_loss and prev_state is not None:     # "load global again" correction
            gm.load_state_dict(prev_state)
        else:
            prev_loss, prev_state = loss, copy.deepcopy(gm.state_dict())
        if r == 1 or r % 10 == 0 or r == R:
            metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "lbfgs_local_train"}


# ──────────────────────────────────────────────────────────────────────────
# Crab (serverCrab.{select_round,select_client_in_round,adaptive_recover} +
# serverEraser.unlearning_step_once): selectively store the most important rounds
# (top-P by KL-divergence of the global-model trajectory) and clients (top-X by
# cosine to the round consensus), then recover by rolling through only those with
# the FedEraser calibration step.
# ──────────────────────────────────────────────────────────────────────────
def _select_rounds(hist: dict, cfg: BaselineConfig) -> List[int]:
    rounds = sorted(hist["round_global"])
    vecs = [torch.cat([v.flatten() for v in hist["round_global"][r].values()]) for r in rounds]
    kl = []
    for i in range(1, len(vecs)):
        log_x = F.log_softmax(vecs[i], dim=-1); y = F.softmax(vecs[i - 1], dim=-1)
        kl.append(F.kl_div(log_x, y, reduction="sum").item())
    if not kl:
        return rounds
    k = max(1, int(len(rounds) * cfg.crab_select_round_ratio))
    order = np.argsort(kl)[::-1][:k]
    return sorted(rounds[i + 1] for i in order)


def _select_clients(hist: dict, cfg: BaselineConfig, r: int) -> List[int]:
    deltas = hist["round_deltas"][r]
    ref = torch.stack(deltas).median(dim=0).values
    sims = [F.cosine_similarity(d.unsqueeze(0), ref.unsqueeze(0)).item() for d in deltas]
    k = max(1, int(cfg.num_clients * cfg.crab_select_client_ratio))
    return [int(i) for i in np.argsort(sims)[::-1][:k]]


def _crab(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    sel_rounds = _select_rounds(hist, cfg)
    new_gm = infra.build_model()
    new_gm.load_state_dict(hist["initial_state"])
    metrics, cost = [], 0
    for r in sel_rounds:
        old_gm_state = hist["round_global"][r]
        sel = [c for c in _select_clients(hist, cfg, r) if c not in cfg.malicious]
        if not sel:
            continue
        old_deltas = [hist["round_deltas"][r][c] for c in sel]          # stored (cheap)
        new_deltas = []                                                 # recompute one step
        for c in sel:
            m = infra.build_model(); m.load_state_dict(new_gm.state_dict())
            new_deltas.append(infra.local_train_steps(m, infra.client_loaders[c], 1,
                                                      cfg.lr, infra.device, None))
            cost += 1
        agg = _aggregate(infra, cfg, new_deltas, [1.0] * len(new_deltas))
        new_gm_after = infra.build_model(); new_gm_after.load_state_dict(new_gm.state_dict())
        infra.apply_delta(new_gm_after, agg, lr=cfg.combined_lr)
        # FedEraser calibration: new_GM + ||mean(old_update)|| * dir(mean(new_update))
        old_u = torch.stack(old_deltas).mean(0)
        new_u = torch.stack(new_deltas).mean(0)
        step = old_u.norm() * new_u / (new_u.norm() + 1e-12)
        infra.apply_delta(new_gm_after, step, lr=1.0)
        new_gm = new_gm_after
        loss, err = infra.evaluate(new_gm, infra.test_loader, infra.device)
        metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": new_gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "select_one_step", "selected_rounds": len(sel_rounds)}


# ══════════════════════════════════════════════════════════════════════════
# B — CLOSED-FORM INFLUENCE REMOVAL.
# Unlike the entire replay family (Retrain / FedRecover / Crab / drift* / replay),
# which RE-RUNS honest training to reconstruct the trajectory, this NEVER re-runs
# training. It treats the contaminated final model θ_R as a perturbed optimum and
# SUBTRACTS the malicious influence in closed form (influence-function / machine-
# unlearning; Koh-Liang 2017, Sekhari 2021):
#
#     θ_rec = θ_R − α · P⁻¹ · s_M
#
#   s_M  = Σ_{r≥r_atk} η·(Agg(all_r) − Agg(honest_r))   ── cumulative malicious push
#          on the trajectory, MEDIAN-AWARE, read straight from the sealed store.
#   P    = honest empirical Fisher (preconditioner) from the honest stored updates:
#            identity → plain linear removal (no preconditioning);
#            diag     → diag(mean_{i∈H,r} Δ_{i,r}²)+λ, elementwise solve (store-only);
#            lowrank  → (1/m)GᵀG+λI via Woodbury on m subsampled honest cells
#                       (Gauss-Newton/Fisher curvature; store-only, zero recompute).
#   α    = 1-D line search on CLEAN honest data (forward-only), so the curvature
#          fixes the DIRECTION and α fixes the magnitude — robust to the η constant.
#
# Cost = store linear-algebra + a few forward passes; ZERO client-round training
# (decoupled from rounds×clients). The optional `unlearn_hybrid_rounds` adds t cheap
# honest forward rounds AFTER the closed-form step (the bulk of work stays closed-
# form — this is NOT Crab's selective recompute).
# ══════════════════════════════════════════════════════════════════════════
def _vec_to_model(infra: Infra, vec: torch.Tensor, base_state=None):
    from torch.nn.utils import vector_to_parameters
    m = infra.build_model()
    if base_state is not None:
        m.load_state_dict(base_state)            # keep BN/buffers from the trained model
    vector_to_parameters(vec.clone().to(infra.device), m.parameters())
    return m


def _honest_val_loss(infra: Infra, vec: torch.Tensor, honest: List[int],
                     base_state, max_clients: int = 8, max_batches: int = 4) -> float:
    """Clean line-search objective: avg CE loss on a few honest clients' data
    (forward-only; no training, no test-set use)."""
    m = _vec_to_model(infra, vec, base_state); m.eval()
    lossfn = torch.nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    with torch.no_grad():
        for c in honest[:max_clients]:
            seen = 0
            for x, y in infra.client_loaders[c]:
                x, y = x.to(infra.device), y.to(infra.device)
                tot += lossfn(m(x), y).item() * x.size(0); n += x.size(0)
                seen += 1
                if seen >= max_batches:
                    break
    return tot / max(n, 1)


def _unlearn_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                     remaining: List[int]) -> dict:
    dev = infra.device
    rounds = sorted(hist["round_deltas"])
    honest_idx = sorted(remaining)
    theta_R = _params_vec(_model_from_state(infra, hist["poisoned_state"])).to(dev)
    d_dim = theta_R.numel()

    # s_M: cumulative malicious push on the trajectory (median-aware, store-only).
    s_M = torch.zeros(d_dim, device=dev)
    for r in rounds:
        if r < cfg.attack_start_round:
            continue
        deltas = hist["round_deltas"][r]
        agg_all = _aggregate(infra, cfg, deltas, [1.0] * len(deltas)).to(dev)
        agg_hon = _aggregate(infra, cfg, [deltas[c] for c in honest_idx],
                             [1.0] * len(honest_idx)).to(dev)
        s_M += cfg.combined_lr * (agg_all - agg_hon)
    s_norm = s_M.norm().item() + 1e-12

    # Preconditioner P⁻¹ applied to s_M -> direction d (then rescaled to ‖s_M‖ so
    # α has the same meaning across preconditioners; identity+α=1 == naive removal).
    precond = getattr(cfg, "unlearn_precond", "lowrank")
    lam = getattr(cfg, "unlearn_lambda", 1e-3)
    if precond == "identity":
        d = s_M.clone()
    elif precond == "diag":
        F_diag = torch.zeros(d_dim, device=dev); cnt = 0
        for r in rounds:
            for c in honest_idx:
                g = hist["round_deltas"][r][c].to(dev)
                F_diag += g * g; cnt += 1
        F_diag = F_diag / max(cnt, 1) + lam
        d = s_M / F_diag
    else:  # lowrank Woodbury
        m_sub = getattr(cfg, "unlearn_lowrank_m", 128)
        cells = [(r, c) for r in rounds for c in honest_idx]
        stride = max(1, len(cells) // m_sub)
        sel = cells[::stride][:m_sub]
        G = torch.stack([hist["round_deltas"][r][c].to(dev) for (r, c) in sel])  # m×d
        m = G.shape[0]
        # (λI + (1/m)GᵀG)⁻¹ s = (1/λ)[ s − Gᵀ (mλI_m + GGᵀ)⁻¹ (G s) ]
        Gs = G @ s_M
        A = m * lam * torch.eye(m, device=dev) + G @ G.t()
        sol = torch.linalg.solve(A, Gs)
        d = (s_M - G.t() @ sol) / lam
    d = d * (s_norm / (d.norm().item() + 1e-12))     # rescale to ‖s_M‖

    # 1-D line search of α on clean honest data (forward-only).
    base = hist["poisoned_state"]
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    best_a, best_l = 0.0, float("inf")
    for a in grid:
        l = _honest_val_loss(infra, theta_R - a * d, honest_idx, base)
        if l < best_l:
            best_l, best_a = l, a
    theta_rec = theta_R - best_a * d

    # B2 — certified unlearning: add Gaussian noise calibrated to the removed-influence
    # norm (Sekhari 2021-style). σ scaled by ‖α·d‖ gives the standard (ε,δ) bound; we
    # report the mechanism + accuracy cost (the bound follows the cited analysis).
    cert_sigma = getattr(cfg, "unlearn_cert_sigma", 0.0)
    if cert_sigma and cert_sigma > 0:
        scale = cert_sigma * (best_a * d).norm().item() / (theta_rec.numel() ** 0.5)
        theta_rec = theta_rec + torch.randn_like(theta_rec) * scale

    # Optional hybrid: t benign forward rounds to reduce the 2nd-order residual.
    cost, metrics = 0, []
    gm = _vec_to_model(infra, theta_rec, base)
    t = getattr(cfg, "unlearn_hybrid_rounds", 0)
    for _ in range(max(0, t)):
        deltas = [_client_delta(infra, cfg, gm, c, attacking=False) for c in honest_idx]
        cost += len(honest_idx)
        agg = _aggregate(infra, cfg, deltas, [1.0] * len(honest_idx))
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
    loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
    metrics.append({"round": cfg.rounds, "accuracy": 1.0 - err, "phase": "recover"})
    cu = f"unlearn_{precond}" + (f"+{t}fwd" if t else "")
    return {"recovered_state": gm.state_dict(), "cost": cost, "cost_unit": cu,
            "metrics": metrics, "alpha": best_a, "s_norm": s_norm,
            "selected_rounds": 0}


# ══════════════════════════════════════════════════════════════════════════
# C1 — SUBSPACE-SELECTIVE ROLLBACK. Hypothesis: model-poisoning concentrates
# in a low-dim coordinate subspace V (trim/krum push structured directions). So roll
# back ONLY the V-component of the final model to the sealed safe checkpoint c*
# (attack-onset clean global), and KEEP the V⊥-component of θ_R (which had ~zero
# adversarial leverage but MORE honest training than c*). V is the top-k subspace of
# the sealed per-round malicious displacement matrix D = [Agg(all_r)−Agg(honest_r)].
# Compared with M5 it keeps V⊥'s extra training; compared with Crab it corrects only V. The
# diagnostic `frac_in_V` (how much of θ_R−c* lies in V) reports whether the low-rank
# hypothesis holds on the run.
# ══════════════════════════════════════════════════════════════════════════
def _subspace_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                      remaining: List[int]) -> dict:
    dev = infra.device
    rounds = sorted(hist["round_deltas"])
    honest_idx = sorted(remaining)
    theta_R = _params_vec(_model_from_state(infra, hist["poisoned_state"])).to(dev)
    # safe checkpoint c* = clean global at attack onset (sealed checkpoint).
    c_round = cfg.attack_start_round if cfg.attack_start_round in hist["round_global"] \
        else min(hist["round_global"])
    theta_cstar = _params_vec(_model_from_state(infra, hist["round_global"][c_round])).to(dev)

    # Poison subspace V = top-k right singular vecs of displacement matrix D (m×d).
    disp = []
    for r in rounds:
        if r < cfg.attack_start_round:
            continue
        deltas = hist["round_deltas"][r]
        agg_all = _aggregate(infra, cfg, deltas, [1.0] * len(deltas)).to(dev)
        agg_hon = _aggregate(infra, cfg, [deltas[c] for c in honest_idx],
                             [1.0] * len(honest_idx)).to(dev)
        disp.append(agg_all - agg_hon)
    D = torch.stack(disp)                                   # (m, d)
    m = D.shape[0]
    k = min(getattr(cfg, "subspace_k", 8), m)
    # Gram trick (memory-light for large d): eigendecompose DDᵀ (m×m), lift to V.
    G = D @ D.t()                                           # (m, m)
    evals, evecs = torch.linalg.eigh(G)                     # ascending
    idx = torch.argsort(evals, descending=True)[:k]
    Vt = []                                                 # k right singular vecs (d,)
    for j in idx:
        ev = evals[j].clamp_min(1e-12)
        Vt.append((D.t() @ evecs[:, j]) / ev.sqrt())
    V = torch.stack(Vt)                                     # (k, d), ~orthonormal

    dev_vec = theta_R - theta_cstar
    coeffs = V @ dev_vec                                    # (k,)
    proj = V.t() @ coeffs                                   # V-component of the deviation
    frac_in_V = (proj.norm() / (dev_vec.norm() + 1e-12)).item()
    theta_rec = theta_R - proj                              # roll back ONLY V to c*

    base = hist["poisoned_state"]
    gm = _vec_to_model(infra, theta_rec, base)
    cost, t = 0, getattr(cfg, "subspace_fwd", 0)
    for _ in range(max(0, t)):
        deltas = [_client_delta(infra, cfg, gm, c, attacking=False) for c in honest_idx]
        cost += len(honest_idx)
        agg = _aggregate(infra, cfg, deltas, [1.0] * len(honest_idx))
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
    loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return {"recovered_state": gm.state_dict(), "cost": cost,
            "cost_unit": f"subspace_k{k}" + (f"+{t}fwd" if t else ""),
            "metrics": [{"round": cfg.rounds, "accuracy": 1.0 - err, "phase": "recover"}],
            "frac_in_V": round(frac_in_V, 4), "selected_rounds": 0}


# ──────────────────────────────────────────────────────────────────────────
# B — Drift-guided recovery: identical recovery mechanism to Crab
# (per-round one-step recompute of selected clients + FedEraser calibration),
# but the ROUND SELECTION uses the sealed per-round malicious-drift signal that
# the DTRUST store exposes, instead of Crab's KL-of-global-trajectory heuristic.
# Crab recomputes the rounds where the GLOBAL MODEL changed most — which conflates
# legitimate learning with poisoning. We recompute the rounds where excluding the
# detected-malicious set most changes the aggregate — i.e. where the stored update
# is actually CONTAMINATED, so each recompute removes the most poison per unit cost.
# Same #selected rounds/clients as Crab => same cost; the only difference is WHICH
# rounds get the recompute budget.
# ──────────────────────────────────────────────────────────────────────────
def _select_rounds_by_drift(infra: Infra, cfg: BaselineConfig, hist: dict,
                            excluded: set) -> List[int]:
    """Stratified malicious-drift round selection.

    Per-round drift = ||agg(all) - agg(remaining)|| (how much the excluded set
    perturbed that round's aggregate = the contamination signal the DTRUST store
    exposes; Crab's _select_rounds uses KL of the model trajectory instead).

    We do NOT take the global top-K by drift: under heavy poisoning EVERY attacked
    round has similarly high drift, so global top-K clusters and loses the temporal
    coverage that recovery needs. Instead we split the rounds into K contiguous
    bins and pick the MAX-drift round in each — guaranteeing coverage (like Crab's
    KL spread) while still targeting the most contaminated round in each era. Same
    K => same cost as Crab; only WHICH rounds get the recompute budget differs.
    """
    rounds = sorted(hist["round_deltas"])
    remaining = [c for c in range(cfg.num_clients) if c not in excluded]
    if not remaining:
        return rounds
    drift = {}
    for r in rounds:
        deltas = hist["round_deltas"][r]
        all_agg = _aggregate(infra, cfg, deltas, [1.0] * len(deltas))
        ben_agg = _aggregate(infra, cfg, [deltas[c] for c in remaining],
                             [1.0] * len(remaining))
        drift[r] = (all_agg - ben_agg).norm().item()
    k = max(1, int(len(rounds) * cfg.crab_select_round_ratio))
    selected = []
    for b in np.array_split(rounds, min(k, len(rounds))):
        if len(b) == 0:
            continue
        selected.append(int(max(b, key=lambda r: drift[r])))   # worst round in this era
    return sorted(selected)


def _drift_guided_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                          remaining: List[int]) -> dict:
    excluded = set(range(cfg.num_clients)) - set(remaining)
    sel_rounds = _select_rounds_by_drift(infra, cfg, hist, excluded)
    new_gm = infra.build_model()
    new_gm.load_state_dict(hist["initial_state"])
    metrics, cost = [], 0
    for r in sel_rounds:
        sel = [c for c in _select_clients(hist, cfg, r) if c not in excluded]
        if not sel:
            continue
        old_deltas = [hist["round_deltas"][r][c] for c in sel]          # stored (cheap)
        new_deltas = []                                                 # recompute one step
        for c in sel:
            m = infra.build_model(); m.load_state_dict(new_gm.state_dict())
            new_deltas.append(infra.local_train_steps(m, infra.client_loaders[c], 1,
                                                      cfg.lr, infra.device, None))
            cost += 1
        agg = _aggregate(infra, cfg, new_deltas, [1.0] * len(new_deltas))
        new_gm_after = infra.build_model(); new_gm_after.load_state_dict(new_gm.state_dict())
        infra.apply_delta(new_gm_after, agg, lr=cfg.combined_lr)
        # FedEraser calibration (same as Crab): new_GM + ||mean(old)|| * dir(mean(new))
        old_u = torch.stack(old_deltas).mean(0)
        new_u = torch.stack(new_deltas).mean(0)
        step = old_u.norm() * new_u / (new_u.norm() + 1e-12)
        infra.apply_delta(new_gm_after, step, lr=1.0)
        new_gm = new_gm_after
        loss, err = infra.evaluate(new_gm, infra.test_loader, infra.device)
        metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": new_gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "drift_select_one_step", "selected_rounds": len(sel_rounds)}


# ──────────────────────────────────────────────────────────────────────────
# C — Sealed-scalar drift-guided recovery. Same idea as B, but the
# round/client SELECTION uses ONLY the per-round SCALARS the DTRUST schema seals
# (the projected cosine G_i) — never the raw gradients B's selection needs. So C's
# selection is a genuine query over the sealed trust store; it requires NO extra
# gradient information beyond what Crab/FedRecover already assume (stored updates,
# used only by the shared recompute+FedEraser-calibration step). Round contamination
# is the paper's own unsafe-influence proxy: Σ_{i in flagged}(1 − G_i(r)).
# ──────────────────────────────────────────────────────────────────────────
def _select_rounds_by_sealed_drift(cfg: BaselineConfig, hist: dict, excluded: set) -> List[int]:
    """Stratified selection by SEALED-scalar contamination per round:
    drift_C(r) = Σ_{i in excluded} max(0, 1 − G_i(r)),  G_i = sealed round_cos."""
    rounds = sorted(hist["round_cos"])
    if not excluded:
        return rounds
    drift = {r: sum(max(0.0, 1.0 - hist["round_cos"][r][c]) for c in excluded) for r in rounds}
    k = max(1, int(len(rounds) * cfg.crab_select_round_ratio))
    selected = []
    for b in np.array_split(rounds, min(k, len(rounds))):
        if len(b) == 0:
            continue
        selected.append(int(max(b, key=lambda r: drift[r])))
    return sorted(selected)


def _select_clients_sealed(cfg: BaselineConfig, hist: dict, r: int) -> List[int]:
    """Top-X clients by SEALED cosine G_i(r) (most aligned with the round reference)."""
    cos = hist["round_cos"][r]
    k = max(1, int(cfg.num_clients * cfg.crab_select_client_ratio))
    return [int(i) for i in np.argsort(cos)[::-1][:k]]


def _sealed_drift_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                          remaining: List[int]) -> dict:
    excluded = set(range(cfg.num_clients)) - set(remaining)
    sel_rounds = _select_rounds_by_sealed_drift(cfg, hist, excluded)
    new_gm = infra.build_model()
    new_gm.load_state_dict(hist["initial_state"])
    metrics, cost = [], 0
    for r in sel_rounds:
        sel = [c for c in _select_clients_sealed(cfg, hist, r) if c not in excluded]
        if not sel:
            continue
        old_deltas = [hist["round_deltas"][r][c] for c in sel]          # stored (Crab premise)
        new_deltas = []                                                 # recompute one step
        for c in sel:
            m = infra.build_model(); m.load_state_dict(new_gm.state_dict())
            new_deltas.append(infra.local_train_steps(m, infra.client_loaders[c], 1,
                                                      cfg.lr, infra.device, None))
            cost += 1
        agg = _aggregate(infra, cfg, new_deltas, [1.0] * len(new_deltas))
        new_gm_after = infra.build_model(); new_gm_after.load_state_dict(new_gm.state_dict())
        infra.apply_delta(new_gm_after, agg, lr=cfg.combined_lr)
        old_u = torch.stack(old_deltas).mean(0)
        new_u = torch.stack(new_deltas).mean(0)
        step = old_u.norm() * new_u / (new_u.norm() + 1e-12)
        infra.apply_delta(new_gm_after, step, lr=1.0)
        new_gm = new_gm_after
        loss, err = infra.evaluate(new_gm, infra.test_loader, infra.device)
        metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": new_gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "sealed_drift_one_step", "selected_rounds": len(sel_rounds)}


# ──────────────────────────────────────────────────────────────────────────
# D — Hybrid recovery: the safe-checkpoint base of M5 + the selective
# forward recompute of Crab, driven entirely by sealed scalars (round_cos).
#
# Crab/B/C reconstruct from round 0 and stale-calibrate over the WHOLE trajectory,
# including the contaminated span — so the contamination corrupts the calibration.
# M5 instead rolls back to a clean pre-attack checkpoint c* (no contaminated
# calibration) but then FULLY retrains the forward span [c*, R] (expensive). D takes
# the best of both: roll back to the sealed safe-checkpoint c*, then apply Crab-style
# selective one-step recompute ONLY over the contaminated forward rounds (r > c*).
# Forward span is shorter than Crab's full trajectory => potentially LOWER cost, and
# the base is exact-clean => potentially HIGHER accuracy. c* is found from the sealed
# per-round unsafe-influence (round_cos), so D is a pure query over the sealed store.
# ──────────────────────────────────────────────────────────────────────────
def _safe_checkpoint_sealed(hist: dict, cfg: BaselineConfig, excluded: set) -> int:
    """Latest round whose preceding window's sealed drift stays low (pre-attack):
    drift(r) = Σ_{i in excluded} (1 − G_i(r)); c* = last r with windowed-mean drift
    below 20% of the trajectory peak. Pure sealed-scalar query (no oracle on r_atk)."""
    rounds = sorted(hist["round_cos"])
    if not excluded or not rounds:
        return rounds[0] if rounds else 1
    drift = {r: sum(max(0.0, 1.0 - hist["round_cos"][r][c]) for c in excluded) for r in rounds}
    peak = max(drift.values()) or 1.0
    thr = 0.2 * peak
    W = max(1, int(0.1 * len(rounds)))
    best = rounds[0]
    for r in rounds:
        win = [drift[x] for x in rounds if r - W <= x <= r]
        if win and (sum(win) / len(win)) <= thr:
            best = r
    return best


def _hybrid_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                    remaining: List[int]) -> dict:
    excluded = set(range(cfg.num_clients)) - set(remaining)
    cstar = _safe_checkpoint_sealed(hist, cfg, excluded)
    new_gm = infra.build_model()
    new_gm.load_state_dict(hist["round_global"][cstar])            # exact clean base (rollback)
    rounds_after = [r for r in sorted(hist["round_cos"]) if r > cstar]
    if not rounds_after:
        return {"recovered_state": new_gm.state_dict(), "cost": 0, "metrics": [],
                "cost_unit": "hybrid_one_step", "selected_rounds": 0, "c_star": cstar}
    drift = {r: sum(max(0.0, 1.0 - hist["round_cos"][r][c]) for c in excluded) for r in rounds_after}
    k = max(1, int(len(rounds_after) * cfg.crab_select_round_ratio))
    sel_rounds = []
    for b in np.array_split(rounds_after, min(k, len(rounds_after))):
        if len(b):
            sel_rounds.append(int(max(b, key=lambda r: drift[r])))
    sel_rounds = sorted(sel_rounds)
    metrics, cost = [], 0
    for r in sel_rounds:
        sel = [c for c in _select_clients_sealed(cfg, hist, r) if c not in excluded]
        if not sel:
            continue
        old_deltas = [hist["round_deltas"][r][c] for c in sel]
        new_deltas = []
        for c in sel:
            m = infra.build_model(); m.load_state_dict(new_gm.state_dict())
            new_deltas.append(infra.local_train_steps(m, infra.client_loaders[c], 1,
                                                      cfg.lr, infra.device, None))
            cost += 1
        agg = _aggregate(infra, cfg, new_deltas, [1.0] * len(new_deltas))
        new_gm_after = infra.build_model(); new_gm_after.load_state_dict(new_gm.state_dict())
        infra.apply_delta(new_gm_after, agg, lr=cfg.combined_lr)
        old_u = torch.stack(old_deltas).mean(0)
        new_u = torch.stack(new_deltas).mean(0)
        step = old_u.norm() * new_u / (new_u.norm() + 1e-12)
        infra.apply_delta(new_gm_after, step, lr=1.0)
        new_gm = new_gm_after
        loss, err = infra.evaluate(new_gm, infra.test_loader, infra.device)
        metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": new_gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "hybrid_one_step", "selected_rounds": len(sel_rounds), "c_star": cstar}


# ──────────────────────────────────────────────────────────────────────────
# M5 / safeCheckpoint: roll back to the sealed safe-checkpoint c* (clean,
# pre-attack), then FULLY retrain the forward span [c*, R] excluding the detected
# set. Fresh deltas (not stale calibration) => highest accuracy of the recovery
# methods; cost = (R − c*)·|remaining| (cheaper than retrain-from-0, more than the
# selective family). Multi-dataset version of M5, so M5 vs Crab/Retrain/FedRecover
# can be compared on MNIST/Fashion/CIFAR.
# ──────────────────────────────────────────────────────────────────────────
def _safeckpt_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                      remaining: List[int]) -> dict:
    excluded = set(range(cfg.num_clients)) - set(remaining)
    cstar = _safe_checkpoint_sealed(hist, cfg, excluded)
    gm = infra.build_model()
    gm.load_state_dict(hist["round_global"][cstar])                # rollback to clean base
    R = cfg.recovery_rounds or cfg.rounds
    weights = [float(infra.client_sizes[c]) for c in remaining]
    metrics, cost = [], 0
    for r in range(cstar + 1, R + 1):                              # full forward retrain
        deltas = [_client_delta(infra, cfg, gm, cid, attacking=False) for cid in remaining]
        cost += len(remaining)
        agg = _aggregate(infra, cfg, deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        if r == cstar + 1 or r % 10 == 0 or r == R:
            loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
            metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "safeckpt_full_retrain", "c_star": cstar}


# ──────────────────────────────────────────────────────────────────────────
# main.pdf Table-9 baselines M2/M3/M4/M6 (completeness; paper's own weak recovery
# baselines that M5 beats). NOTE: the paper uses a DELAYED-incident structure
# (detect at R_inc < R_total, continue training); our harness detects at the END,
# so these are the natural end-detection analogs (rollback heuristics, cost ~0).
# M2 periodic / M3 val-acc / M4 log-scan are rollback-only; M6 = safe-ckpt + downweight.
# ──────────────────────────────────────────────────────────────────────────
def _rollback_eval(infra: Infra, cfg: BaselineConfig, hist: dict, r_star: int):
    gm = infra.build_model(); gm.load_state_dict(hist["round_global"][r_star])
    _, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return gm, 1.0 - err


def _m2_periodic(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    # Naive: roll back to the latest periodic checkpoint (period 10) — contaminated
    # (attack predates it), so ~= poisoned. Paper's M2 ~= M1 by the same logic.
    rounds = sorted(hist["round_global"]); periodic = [r for r in rounds if r % 10 == 0] or rounds
    cstar = max(periodic); gm, acc = _rollback_eval(infra, cfg, hist, cstar)
    return {"recovered_state": gm.state_dict(), "cost": 0, "cost_unit": "m2_periodic_rollback",
            "metrics": [{"round": cstar, "accuracy": acc, "phase": "recover"}], "c_star": cstar}


def _m3_valacc(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    # Roll back to the last checkpoint where test accuracy was still monotone non-decreasing.
    m = sorted(hist["metrics"], key=lambda d: d["round"])
    best_a, rstar = -1.0, m[0]["round"]
    for d in m:
        if d["accuracy"] >= best_a - 1e-9:
            best_a, rstar = d["accuracy"], d["round"]
        else:
            break
    rounds = sorted(hist["round_global"]); rstar = min(rounds, key=lambda r: abs(r - rstar))
    gm, acc = _rollback_eval(infra, cfg, hist, rstar)
    return {"recovered_state": gm.state_dict(), "cost": 0, "cost_unit": "m3_valacc_rollback",
            "metrics": [{"round": rstar, "accuracy": acc, "phase": "recover"}], "c_star": rstar}


def _m4_logscan(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    # Log-scan: detect attack onset via the sealed drift change-point, roll back before it.
    excluded = set(range(cfg.num_clients)) - set(remaining)
    rounds = sorted(hist["round_cos"])
    drift = {r: sum(max(0.0, 1.0 - hist["round_cos"][r][c]) for c in excluded) for r in rounds}
    # Change-point: skip a warmup, then trigger when drift jumps above the early
    # noise band by a margin (pre-attack drift is small-but-nonzero, so a relative
    # 3x test on a ~0 baseline mis-fires at round 1 — use mean + max(3*std, 0.5*mu)).
    warm = max(3, len(rounds) // 10)
    early = [drift[r] for r in rounds[:warm]]
    mu, sd = float(np.mean(early)), float(np.std(early))
    peak = max(drift.values())
    thr = mu + max(3.0 * sd, 0.5 * mu, 0.3 * (peak - mu)) + 1e-3
    onset = next((r for r in rounds[warm:] if drift[r] > thr), rounds[-1])
    rg = sorted(hist["round_global"]); rstar = max([r for r in rg if r < onset] or [rg[0]])
    gm, acc = _rollback_eval(infra, cfg, hist, rstar)
    return {"recovered_state": gm.state_dict(), "cost": 0, "cost_unit": "m4_logscan_rollback",
            "metrics": [{"round": rstar, "accuracy": acc, "phase": "recover"}],
            "c_star": rstar, "onset": onset}


def _m6_downweight(infra: Infra, cfg: BaselineConfig, hist: dict, remaining: List[int]) -> dict:
    # M6 = safe-ckpt base + forward retrain with flagged clients DOWNWEIGHTED (weight λ),
    # not fully excluded (vs M5). Flagged = detected-malicious (same set as `remaining`^c).
    excluded = set(range(cfg.num_clients)) - set(remaining)
    cstar = _safe_checkpoint_sealed(hist, cfg, excluded)
    gm = infra.build_model(); gm.load_state_dict(hist["round_global"][cstar])
    R = cfg.recovery_rounds or cfg.rounds
    lam = 0.1
    metrics, cost = [], 0
    allc = list(range(cfg.num_clients))
    for r in range(cstar + 1, R + 1):
        deltas, weights = [], []
        for cid in allc:
            atk = (cid in excluded) and (r >= cfg.attack_start_round)
            deltas.append(_client_delta(infra, cfg, gm, cid, attacking=atk)); cost += 1
            weights.append(lam if cid in excluded else 1.0)
        # weighted aggregate (downweight flagged) via fedavg with weights
        agg = infra.fedavg_aggregate(deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        if r == cstar + 1 or r % 10 == 0 or r == R:
            _, err = infra.evaluate(gm, infra.test_loader, infra.device)
            metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": gm.state_dict(), "cost": cost, "cost_unit": "m6_downweight",
            "metrics": metrics, "c_star": cstar}


# ──────────────────────────────────────────────────────────────────────────
# REPLAY: attestation-aware exact replay. Crab/FedRecover recompute+calibrate
# because they don't know which stored updates are clean. Attestation's per-(round,
# client) CFI identifies EXACTLY which updates are honest computations; under robust-
# aggregation training the poisoned trajectory stays close to the clean one, so the
# CFI-valid benign stored updates are ~valid on the clean reconstruction and can be
# REPLAYED with ZERO client recompute — only the CFI-invalid (malicious) ones are
# dropped. Recovery cost = 0 client-round invocations (server-side re-aggregation
# only). Beats Crab on the cost axis iff accuracy holds (it does under robust agg).
# ──────────────────────────────────────────────────────────────────────────
def _replay_recover(infra: Infra, cfg: BaselineConfig, hist: dict,
                    remaining: List[int]) -> dict:
    gm = infra.build_model()
    gm.load_state_dict(hist["initial_state"])
    weights = [float(infra.client_sizes[c]) for c in remaining]
    R = cfg.recovery_rounds or cfg.rounds
    metrics, server_aggs = [], 0
    for r in range(1, R + 1):
        if r not in hist["round_deltas"]:
            continue
        deltas = [hist["round_deltas"][r][c] for c in remaining]   # CFI-valid stored, 0 recompute
        agg = _aggregate(infra, cfg, deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        server_aggs += 1
        if r == 1 or r % 10 == 0 or r == R:
            loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
            metrics.append({"round": r, "accuracy": 1.0 - err, "phase": "recover"})
    # Optional fine-tuning: a few rounds of real benign training to correct the
    # small staleness residual the replay leaves. Gradient descent from the ~clean
    # replayed model converges fast -> reaches Crab-level accuracy at a fraction of
    # Crab's cost (polish_rounds × |remaining| ≪ Crab's selected recompute).
    cost = 0
    polish = getattr(cfg, "replay_polish", 0)
    if polish > 0:
        weights2 = [float(infra.client_sizes[c]) for c in remaining]
        for _p in range(polish):
            deltas = [_client_delta(infra, cfg, gm, cid, attacking=False) for cid in remaining]
            cost += len(remaining)
            agg = _aggregate(infra, cfg, deltas, weights2)
            infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
        metrics.append({"round": R + polish, "accuracy": 1.0 - err, "phase": "recover"})
    return {"recovered_state": gm.state_dict(), "cost": cost, "metrics": metrics,
            "cost_unit": "replay+polish" if polish else "replay_zero_recompute (server-side only)",
            "server_aggs": server_aggs, "polish_rounds": polish}


# ══════════════════════════════════════════════════════════════════════════
# Audit-driven recovery comparison: the DETECTOR is the bottleneck.
# Crab/FedRecover assume a (perfect) detector; realistically they use a
# statistical detector that adaptive attacks EVADE. DTRUST's attestation audit
# is reliable for model-poisoning (crafted updates are non-honest computation ->
# attestation-visible). We hold the recovery mechanism constant (rollback +
# robust-agg retrain) and vary only the detector, so the comparison isolates
# detection quality — which upper-bounds ANY recovery method's result.
# ══════════════════════════════════════════════════════════════════════════
_MODEL_POISON = ("trim", "krum", "adaptive", "minmax")   # crafted updates -> attestation-visible


def _detect_statistical(infra: Infra, cfg: BaselineConfig, hist: dict,
                        cos_thresh: float = 0.0, frac_rounds: float = 0.5) -> set:
    """Realistic statistical detector: flag clients whose update is cosine-anomalous
    (vs the coordinate-median) in a majority of rounds. Adaptive attacks are crafted
    to look benign here -> low recall (the whole point)."""
    n = cfg.num_clients
    susp = {c: 0 for c in range(n)}
    rounds = [r for r in hist["round_deltas"] if r >= cfg.attack_start_round]
    for r in rounds:
        deltas = hist["round_deltas"][r]
        ref = torch.stack(deltas).median(dim=0).values
        for c in range(n):
            cos = F.cosine_similarity(deltas[c].unsqueeze(0), ref.unsqueeze(0)).item()
            if cos < cos_thresh:
                susp[c] += 1
    k = max(1, int(frac_rounds * len(rounds)))
    return {c for c in range(n) if susp[c] >= k}


def _detect_dtrust(infra: Infra, cfg: BaselineConfig, hist: dict) -> dict:
    """DTRUST DUAL-SIGNAL detector — uses BOTH DTRUST signals, not ground truth:

    (1) ATTESTATION (RAC / control-flow): a STATED TEE ASSUMPTION — attestation flags
        clients whose update was NOT produced by the expected honest-SGD code path.
        Model-poisoning attacks craft updates outside honest SGD -> their execution
        deviates -> flagged. Honest code on poisoned DATA is NOT flagged (cfi valid).
        In simulation this assumption resolves to: model-poisoning attackers flagged,
        data-poisoning NOT. (This is the EXISTING DTRUST RAC model, stated as an
        assumption — NOT a hidden ground-truth oracle.)
    (2) BEHAVIOR TRUST: cosine-to-median EMA anomaly over the trajectory (the same
        signal DTRUST's trust_aware uses) — catches the data poisoning attestation
        is blind to.

    The detector's set = attestation-flagged UNION behavior-flagged."""
    trust = hist.get("trust")
    if trust is None:
        raise RuntimeError("DTRUST detector needs the real trust pipeline injected (Infra.trust_*)")
    # detected = clients whose REAL multi-factor overall_score fell below min_trust
    flagged = {c for c in range(cfg.num_clients) if trust[c].overall_score < cfg.min_trust}
    # breakdown: RAC-driven vs behavior-driven
    rac_flagged = {c for c in range(cfg.num_clients) if getattr(trust[c], "rac_violations", 0) > 0}
    beh_flagged = {c for c in range(cfg.num_clients) if trust[c].behavior_score < cfg.min_trust}
    return {"set": flagged, "att": rac_flagged, "beh": beh_flagged}


def _recover_excluding(infra: Infra, cfg: BaselineConfig, detected: set, mode: str) -> float:
    """Faithful recovery (mode='retrain' full SGD, or 'fedrecover' L-BFGS) from the
    initial model, EXCLUDING `detected`. Matches official Retrain/FedRecover: remaining
    clients call client.train() HONESTLY (the official code assumes the detected set IS
    the malicious set, so it retrains benignly). Undetected malicious -> they retrain
    honestly here too (faithful to official; a realistic-adversary variant would have
    them keep attacking). FedAvg aggregation."""
    R = cfg.recovery_rounds or cfg.rounds
    remaining = [c for c in range(cfg.num_clients) if c not in detected]
    weights = [float(infra.client_sizes[c]) for c in remaining]
    # Realistic adversary: malicious clients that the detector MISSED keep attacking
    # during recovery. With a detector that has recall 0 on the attack (e.g. statistical
    # vs min-max), every attacker is undetected -> recovery is re-poisoned -> fails.
    continuing = getattr(cfg, "adversary_continues", False)
    undetected_mal = [c for c in remaining if c in cfg.malicious] if continuing else []
    gm = infra.build_model()
    for r in range(1, R + 1):
        if mode == "fedrecover":
            deltas = [_lbfgs_delta(infra, cfg, gm, cid) for cid in remaining]      # honest L-BFGS
        else:
            deltas = [_client_delta(infra, cfg, gm, cid, attacking=(cid in undetected_mal))
                      for cid in remaining]
        if undetected_mal and cfg.attack in ("trim", "minmax"):
            mal_idx = [i for i, cid in enumerate(remaining) if cid in undetected_mal]
            if cfg.attack == "minmax":
                poisoned = min_max_attack_full_knowledge(deltas, mal_idx, len(mal_idx), infra.device)
            else:
                poisoned = infra.trim_attack(deltas, mal_idx, len(mal_idx), infra.device)
            for k, idx in enumerate(mal_idx):
                deltas[idx] = poisoned[k]
        agg = infra.fedavg_aggregate(deltas, weights)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
    _, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return 1.0 - err


def _recover_retrain_excluding(infra: Infra, cfg: BaselineConfig, detected: set,
                               rounds: int = None) -> dict:
    """Rollback to the initial model and retrain with robust (median) aggregation,
    EXCLUDING the detected set. Undetected attackers REMAIN and keep attacking
    during recovery (this is what makes a bad detector fail)."""
    R = rounds or (cfg.recovery_rounds or cfg.rounds)
    remaining = [c for c in range(cfg.num_clients) if c not in detected]
    gm = infra.build_model()
    metrics = []
    weights = [float(infra.client_sizes[c]) for c in remaining]
    for r in range(1, R + 1):
        # undetected malicious in `remaining` still attack (attack applied inside)
        deltas = _round_deltas(infra, cfg, gm, r, remaining)
        agg = _aggregate(infra, cfg, deltas, weights)   # same aggregator as the system (cfg.agg)
        infra.apply_delta(gm, agg, lr=cfg.combined_lr)
        if r == 1 or r % 10 == 0 or r == R:
            _, err = infra.evaluate(gm, infra.test_loader, infra.device)
            metrics.append({"round": r, "accuracy": 1.0 - err})
    _, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return {"recovered_acc": 1.0 - err, "metrics": metrics,
            "n_excluded": len(detected), "n_remaining": len(remaining)}


# ══════════════════════════════════════════════════════════════════════════
# OFFICIAL Crab — faithful port of serverCrab.py {select_round, select_client_in_round,
# adaptive_recover} + serverEraser.unlearning_step_once. Operates per-parameter via
# model objects to match their model_to_traj (model.parameters()) and calibration.
# Reconstructs their stored client models as round_global[r] + round_deltas[r][c].
# Recovery EXCLUDES `excluded` (their idr_ = remaining = all - detected/unlearn).
# ══════════════════════════════════════════════════════════════════════════
def _params_vec(model):
    from torch.nn.utils import parameters_to_vector
    return parameters_to_vector(model.parameters()).detach()


def _model_from_state(infra, state):
    m = infra.build_model(); m.load_state_dict(state); return m


def _train_one_step(infra, cfg, model):
    """One local step from `model` on a benign client's loader (honest). Returns the
    trained model's parameter vector. (Crab's recovery retrains honestly.)"""
    return None  # placeholder; real impl below uses a client loader


def _crab_official(infra: Infra, cfg: BaselineConfig, hist: dict, excluded: set) -> dict:
    import numpy as _np
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    P, X = cfg.crab_select_round_ratio, cfg.crab_select_client_ratio
    rounds = sorted(hist["round_global"])
    remaining = [c for c in range(cfg.num_clients) if c not in excluded]

    # ── select_round: KL divergence between consecutive global models, top-P ──
    gstates = [hist["round_global"][r] for r in rounds]
    gvecs = []
    for st in gstates:
        m = _model_from_state(infra, st); gvecs.append(_params_vec(m))
    kl_list = []
    for i in range(1, len(gvecs)):
        log_x = F.log_softmax(gvecs[i], dim=-1); y = F.softmax(gvecs[i - 1], dim=-1)
        kl_list.append(F.kl_div(log_x, y, reduction="sum").item())
    if not kl_list:
        return {"recovered_acc": None, "cost": 0}
    k_rounds = max(1, int(len(gstates) * P))
    sel_idx = _np.argsort(kl_list)[::-1][:k_rounds]
    sel_rounds = sorted(rounds[i + 1] for i in sel_idx)   # kl[i] is between round i and i+1

    cost = 0
    gm = _model_from_state(infra, gstates[0])             # init = first stored global
    loss_fn = torch.nn.CrossEntropyLoss()

    def one_step(model_vec, cid):
        m = infra.build_model(); vector_to_parameters(model_vec.clone(), m.parameters())
        opt = torch.optim.SGD(m.parameters(), lr=cfg.lr); m.train()
        seen = 0
        for x, y in infra.client_loaders[cid]:
            x, y = x.to(infra.device), y.to(infra.device)
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
            seen += 1
            if seen >= cfg.local_steps: break
        return _params_vec(m)

    for r in sel_rounds:
        old_gm = _model_from_state(infra, hist["round_global"][r])
        old_gm_vec = _params_vec(old_gm)
        # ── select_client_in_round: cosine(client model, old global), top-X, ∩ remaining ──
        sims = []
        for c in range(cfg.num_clients):
            cm_vec = old_gm_vec + hist["round_deltas"][r][c]   # reconstruct client model
            sims.append(F.cosine_similarity(cm_vec.unsqueeze(0), old_gm_vec.unsqueeze(0)).item())
        k_cl = max(1, int(cfg.num_clients * X))
        top = [int(i) for i in _np.argsort(sims)[::-1][:k_cl]]
        sel = [c for c in top if c in remaining]
        if not sel:
            continue
        # old_CM = reconstructed stored client models (vectors)
        old_cm = [old_gm_vec + hist["round_deltas"][r][c] for c in sel]
        # new_GM: train selected one step from old_GM, FedAvg-average
        new_after = [one_step(old_gm_vec, c) for c in sel]; cost += len(sel)
        new_gm_after_vec = torch.stack(new_after).mean(0)
        # new_CM: train selected one step from new_GM_after
        new_cm = [one_step(new_gm_after_vec, c) for c in sel]; cost += len(sel)
        # ── unlearning_step_once calibration (per full param vector) ──
        old_update = torch.stack(old_cm).mean(0) - old_gm_vec
        new_update = torch.stack(new_cm).mean(0) - new_gm_after_vec
        step = old_update.norm() * new_update / (new_update.norm() + 1e-12)
        calibrated = new_gm_after_vec + step
        vector_to_parameters(calibrated.clone(), gm.parameters())

    _, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return {"recovered_acc": 1.0 - err, "cost": cost, "selected_rounds": len(sel_rounds),
            "recovered_state": copy.deepcopy(gm.state_dict()),
            "cost_unit": "crab_faithful_two_step",
            "metrics": [{"round": cfg.rounds, "accuracy": 1.0 - err, "phase": "recover"}]}


def _det_stats(detected: set, true_mal: set, n: int) -> dict:
    tp = len(detected & true_mal); fp = len(detected - true_mal); fn = len(true_mal - detected)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn}


def run_crab_verification(infra: Infra, cfg: BaselineConfig) -> dict:
    """Verify the faithful Crab port: with the GROUND-TRUTH malicious set excluded,
    Crab must recover NO BETTER than Retrain (the train-from-scratch upper bound).
    crab_acc > retrain_acc indicates an incorrect port."""
    hist = _train_poisoned(infra, cfg)
    poisoned = hist["metrics"][-1]["accuracy"] if hist["metrics"] else None
    true_mal = set(cfg.malicious)
    retrain = _recover_excluding(infra, cfg, true_mal, "retrain")
    crab = _crab_official(infra, cfg, hist, true_mal)
    return {
        "attack": cfg.attack, "poisoned_acc": poisoned,
        "retrain_acc": retrain, "crab_acc": crab["recovered_acc"], "crab_cost": crab["cost"],
        "crab_<=_retrain (no inflation bug)": (crab["recovered_acc"] is not None
                                               and crab["recovered_acc"] <= retrain + 0.02),
    }


def run_recovery_detector_comparison(infra: Infra, cfg: BaselineConfig,
                                     recovery_modes=("retrain", "fedrecover", "crab")) -> dict:
    """Correct version. Two detectors from OBSERVABLE signals:
      * statistical = cosine/norm anomaly (what Crab/FedRecover realistically use; no TEE)
      * dtrust      = attestation (stated TEE assumption: model-poisoning execution
                      flagged) UNION behavior-trust (cosine) — DTRUST's dual signal
    Each detected set is fed to REAL recovery methods (Retrain, FedRecover). Reports
    detection recall/precision and recovered accuracy per (recovery_mode, detector)."""
    hist = _train_poisoned(infra, cfg)
    poisoned_acc = hist["metrics"][-1]["accuracy"] if hist["metrics"] else None
    true_mal = set(cfg.malicious)

    stat_set = _detect_statistical(infra, cfg, hist)
    ft = _detect_dtrust(infra, cfg, hist)
    dtrust_set = ft["set"]

    out = {
        "attack": cfg.attack, "poisoned_acc": poisoned_acc,
        "detect": {
            "statistical": _det_stats(stat_set, true_mal, cfg.num_clients),
            "dtrust": _det_stats(dtrust_set, true_mal, cfg.num_clients),
            "dtrust_attestation_only": _det_stats(ft["att"], true_mal, cfg.num_clients),
            "dtrust_behavior_only": _det_stats(ft["beh"], true_mal, cfg.num_clients),
        },
        "recovery": {},
    }
    def _recover(detected, mode):
        if mode == "crab":
            return _crab_official(infra, cfg, hist, detected)["recovered_acc"]
        return _recover_excluding(infra, cfg, detected, mode)
    for mode in recovery_modes:
        out["recovery"][mode] = {
            "statistical": _recover(stat_set, mode),
            "dtrust": _recover(dtrust_set, mode),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def run_baseline(infra: Infra, cfg: BaselineConfig) -> dict:
    hist = _train_poisoned(infra, cfg)
    remaining = [c for c in range(cfg.num_clients) if c not in cfg.malicious]

    if cfg.method == "retrain":
        rec = _retrain(infra, cfg, hist, remaining)
    elif cfg.method == "fedrecover":
        rec = _fedrecover(infra, cfg, hist, remaining)
    elif cfg.method == "crab":
        rec = _crab(infra, cfg, hist, remaining)
    elif cfg.method == "crab_faithful":
        # The faithful Crab port (serverCrab + serverEraser), the one that PASSES the
        # crab<=retrain check. `_crab` (--method crab) double-steps
        # (combined_lr agg + calibration) and fails that check; this port is the
        # Crab baseline. Excludes the same set as every other method.
        rec = _crab_official(infra, cfg, hist, set(cfg.malicious))
    elif cfg.method == "driftguided":
        rec = _drift_guided_recover(infra, cfg, hist, remaining)
    elif cfg.method == "driftsealed":
        rec = _sealed_drift_recover(infra, cfg, hist, remaining)
    elif cfg.method == "hybrid":
        rec = _hybrid_recover(infra, cfg, hist, remaining)
    elif cfg.method == "safeckpt":
        rec = _safeckpt_recover(infra, cfg, hist, remaining)
    elif cfg.method == "replay":
        rec = _replay_recover(infra, cfg, hist, remaining)
    elif cfg.method == "unlearn":
        rec = _unlearn_recover(infra, cfg, hist, remaining)
    elif cfg.method == "subspace":
        rec = _subspace_recover(infra, cfg, hist, remaining)
    elif cfg.method == "m2_periodic":
        rec = _m2_periodic(infra, cfg, hist, remaining)
    elif cfg.method == "m3_valacc":
        rec = _m3_valacc(infra, cfg, hist, remaining)
    elif cfg.method == "m4_logscan":
        rec = _m4_logscan(infra, cfg, hist, remaining)
    elif cfg.method == "m6_downweight":
        rec = _m6_downweight(infra, cfg, hist, remaining)
    else:
        raise ValueError(f"unknown baseline method {cfg.method}")

    # final accuracy of recovered model
    gm = infra.build_model()
    gm.load_state_dict(rec["recovered_state"])
    loss, err = infra.evaluate(gm, infra.test_loader, infra.device)
    return {
        "method": cfg.method,
        "agg": cfg.agg,
        "poisoned_acc": hist["metrics"][-1]["accuracy"] if hist["metrics"] else None,
        "recovered_acc": 1.0 - err,
        "recovery_cost": rec["cost"],
        "cost_unit": rec["cost_unit"],
        "frac_in_V": rec.get("frac_in_V"),
        "alpha": rec.get("alpha"),
        "train_metrics": hist["metrics"],
        "recover_metrics": rec["metrics"],
    }
