"""Policy-revision sensitivity surface over (attack strength lf_flip_frac × decay).

Instead of one operating point, report the
WHOLE (lf_flip_frac, decay) grid, so the "v1 misses / v2 catches" regime is a
*region*, not a single tuned cell. The attack-strength axis reuses existing FL
dumps; the decay axis is swept fully OFFLINE (decay is an audit-side knob), so
no new training is needed for cells whose dump already exists.

For each cell we report, from the stateful replay (seed 0 unless --seeds given):
  recall_v1, recall_v2, mean-exclusion-earlier, decision flips, extra honest.

    python -m experiment_scripts.dtrust_experiments.c1_audit.sensitivity_surface \
        --dumps 0.02:results/recovery/c1_lf_p0.02 0.03:results/recovery/c1_final_p0.03 \
                0.05:results/recovery/c1_lf_p0.05 0.10:results/recovery/c1_lf_p0.10 \
                0.15:results/recovery/c1_lf_p0.15 0.25:results/recovery/c1_lf_p0.25 \
        --decays 0.85 0.90 0.95 0.98 --out results/recovery/c1_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .policies import policy_pair, load_base_config
from .real_event import evidence_from_dump
from .audit import differential_audit_stateful


def _cell(dump_dir, seed, decay, h_v1):
    base = load_base_config()
    v1, v2 = policy_pair(base, h_v1=h_v1, decay=decay)
    ev, mal, meta = evidence_from_dump(Path(dump_dir) / f"dump_seed{seed}.npz", v1, seed=seed)
    diff = differential_audit_stateful(ev, v1, v2, mal, meta["r_atk"])
    return {
        "recall_v1": diff["v1"]["recall"], "recall_v2": diff["v2"]["recall"],
        "flips": diff["flip_total"],
        "mean_excl_earlier": diff["mean_exclusion_rounds_earlier"] or 0.0,
        "extra_honest": diff["extra_honest_quarantined_v2"],
        "extra_malicious": diff["extra_malicious_caught_v2"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="entries 'flip_frac:dump_dir'")
    ap.add_argument("--decays", type=float, nargs="+", default=[0.85, 0.90, 0.95, 0.98])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--h_v1", type=float, default=15.0)
    ap.add_argument("--out", default="results/recovery/c1_sensitivity")
    args = ap.parse_args()

    fracs, dirs = [], []
    for d in args.dumps:
        ff, path = d.split(":", 1)
        fracs.append(float(ff)); dirs.append(path)
    order = np.argsort(fracs)
    fracs = [fracs[i] for i in order]; dirs = [dirs[i] for i in order]

    metrics = ["recall_v1", "recall_v2", "mean_excl_earlier", "flips", "extra_honest"]
    grid = {m: np.full((len(args.decays), len(fracs)), np.nan) for m in metrics}
    table = []
    for j, (ff, dd) in enumerate(zip(fracs, dirs)):
        for i, dec in enumerate(args.decays):
            vals = [_cell(dd, s, dec, args.h_v1) for s in args.seeds]
            agg = {m: float(np.mean([v[m] for v in vals])) for m in metrics}
            for m in metrics:
                grid[m][i, j] = agg[m]
            table.append({"flip_frac": ff, "decay": dec, **agg})

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"flip_fracs": fracs, "decays": args.decays, "cells": table},
              open(out / "sensitivity.json", "w"), indent=2)

    # Heatmaps: recall_v1 (lower=more v1 misses), v1-v2 recall gap, mean-excl-earlier.
    panels = [("recall_v1", "v1 recall (lower = v1 misses more)", "Reds_r"),
              ("mean_excl_earlier", "v2 mean earlier-exclusion (rounds)", "Greens"),
              ("flips", "admit->quarantine flips", "Blues")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (m, title, cmap) in zip(axes, panels):
        im = ax.imshow(grid[m], aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(fracs))); ax.set_xticklabels([f"{f:.2f}" for f in fracs])
        ax.set_yticks(range(len(args.decays))); ax.set_yticklabels([f"{d:.2f}" for d in args.decays])
        ax.set_xlabel("attack strength  lf_flip_frac"); ax.set_ylabel("decay")
        ax.set_title(title, fontsize=10)
        for i in range(len(args.decays)):
            for j in range(len(fracs)):
                ax.text(j, i, f"{grid[m][i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("C1 sensitivity surface (seed-mean): the v1-misses / v2-catches regime",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "c1_sensitivity.png", dpi=150)
    print(f"wrote {out/'c1_sensitivity.png'} and sensitivity.json  "
          f"({len(fracs)} flip_fracs × {len(args.decays)} decays)")


if __name__ == "__main__":
    main()
