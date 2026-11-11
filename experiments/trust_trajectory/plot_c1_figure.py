"""Policy-revision figure (2 panels) from a sealed delayed-compromise dump.

Panel A -- mechanism: the SPRT trust trajectory under the shared dynamics. The
  malicious median trust descends into and SITS in the [tau_v1, tau_v2] gap band
  (admitted by the lenient online policy rho_v1, quarantined by the audit policy
  rho_v2) while honest trust stays high: the clients are acceptable under rho_v1
  and not under rho_v2.
Panel B -- consequence: per round, how many decisions flip admit->quarantine
  when rho_v2 is replayed (malicious caught vs honest collateral), with R_atk,
  R_inc and the two safe-checkpoints c*(v1) / c*(v2).

    python -m experiment_scripts.dtrust_experiments.c1_audit.plot_c1_figure \
        --dump results/recovery/c1_final_p0.03/dump_seed0.npz \
        --h_v1 15 --decay 0.95 --out fig_c1.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .policies import policy_pair, load_base_config
from .real_event import evidence_from_dump
from .sprt_replay import replay
from .audit import safe_checkpoint_drift, load_sealed
from .seal import seal_event


C_MAL, C_HON, C_GAP = "#c0392b", "#2c7fb8", "#bdbf3f"
C_ATK, C_INC, C_C1, C_C2 = "#e67e22", "#16557a", "#27ae60", "#8e44ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--h_v1", type=float, default=15.0)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--tau_v1", type=float, default=0.30)
    ap.add_argument("--tau_v2", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load_base_config()
    theta_v1, theta_v2 = policy_pair(base, h_v1=args.h_v1, decay=args.decay)
    m = re.search(r"seed(\d+)", Path(args.dump).stem)
    seed = int(m.group(1)) if m else 0           # match the audit's blind-att RNG seed
    ev, mal_ids, meta = evidence_from_dump(args.dump, theta_v1, seed=seed)
    r_atk, r_inc = meta["r_atk"], meta["r_inc"]
    R, N = ev.n_rounds, ev.n_participants
    M = set(mal_ids); hon = [i for i in range(N) if i not in M]
    rounds = np.arange(R)

    d1 = replay(ev, theta_v1)
    d2 = replay(ev, theta_v2)
    ts = d1.t_score                                   # policy-independent trust

    # Per-round flip counts (admitted by v1, banned by v2).
    flip = d1.admitted & ~d2.admitted
    flip_mal = np.array([flip[r, list(M)].sum() for r in range(R)])
    flip_hon = np.array([flip[r, hon].sum() for r in range(R)])

    # Safe-checkpoint c* from the sealed ledger (deliverable row).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        seal_event(ev, d1, theta_v1, td, 0, r_atk, r_inc, mal_ids)
        handle = load_sealed(td, 0)
        q9 = safe_checkpoint_drift(handle, theta_v1, theta_v2, r_inc, 20, 10, 0.01)
    c1, c2 = q9["c_star_v1"], q9["c_star_v2"]

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})

    # ---- Panel A: trust trajectories + gap band ----
    axA.axhspan(args.tau_v1, args.tau_v2, color=C_GAP, alpha=0.30, zorder=0)
    for tau, c, lab in [(args.tau_v1, C_C1, f"τ_min(v1)={args.tau_v1:.2f}"),
                        (args.tau_v2, C_C2, f"τ_min(v2)={args.tau_v2:.2f}")]:
        axA.axhline(tau, ls="--", lw=1.3, c=c, label=lab)

    def _smooth(x, w=7):
        if len(x) < w:
            return x
        k = np.ones(w) / w
        return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, "valid")[:len(x)]

    def band(idx, color, lab):
        arr = ts[:, idx]
        med = _smooth(np.median(arr, axis=1))
        lo = _smooth(np.percentile(arr, 25, axis=1))
        hi = _smooth(np.percentile(arr, 75, axis=1))
        axA.fill_between(rounds, lo, hi, color=color, alpha=0.16, zorder=1, lw=0)
        axA.plot(rounds, med, color=color, lw=2.4, label=lab, zorder=3)

    band(hon, C_HON, "honest trust (median, IQR)")
    band(list(M), C_MAL, "malicious trust (median, IQR)")
    for x, c in [(r_atk, C_ATK), (r_inc, C_INC)]:
        axA.axvline(x, ls=":", lw=1.4, c=c, zorder=2)
    axA.set_ylabel("SPRT trust  T_i = σ(λ_i)")
    axA.set_ylim(0, 1)
    axA.set_title("C1: stealthy trust sits in the ρ(v1)/ρ(v2) gap — admitted online, "
                  "quarantined on audit")
    axA.annotate("malicious trust lives in the gap:\nadmitted by v1, quarantined by v2",
                 xy=((r_atk + r_inc) / 2, (args.tau_v1 + args.tau_v2) / 2),
                 xytext=((r_atk + r_inc) / 2 + 6, 0.66), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#555"))
    h = [Patch(facecolor=C_GAP, alpha=0.3, label="gap band [τ(v1), τ(v2)]")]
    axA.legend(handles=h + axA.get_legend_handles_labels()[0],
               loc="center left", fontsize=8, framealpha=0.92)

    # ---- Panel B: per-round decision flips ----
    axB.fill_between(rounds, 0, flip_mal, color=C_MAL, alpha=0.85, step="mid",
                     label=f"malicious caught by v2 (n={int(flip_mal.sum())})")
    axB.fill_between(rounds, flip_mal, flip_mal + flip_hon, color="#95a5a6",
                     alpha=0.9, step="mid",
                     label=f"honest collateral (n={int(flip_hon.sum())})")
    for x, c, lab in [(r_atk, C_ATK, f"R_atk={r_atk}"), (r_inc, C_INC, f"R_inc={r_inc}"),
                      (c1, C_C1, f"c*(v1)={c1}"), (c2, C_C2, f"c*(v2)={c2}")]:
        if x is not None:
            axB.axvline(x, ls="--", lw=1.3, c=c, label=lab)
    axB.set_xlabel("round"); axB.set_ylabel("# admit→quarantine\nflips this round")
    axB.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.92)
    axB.set_xlim(0, R - 1)

    fig.tight_layout()
    out = args.out or str(Path(args.dump).with_name("c1_figure.png"))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  | flips: {int(flip_mal.sum())} malicious + "
          f"{int(flip_hon.sum())} honest | c*(v1)={c1} c*(v2)={c2}")


if __name__ == "__main__":
    main()
