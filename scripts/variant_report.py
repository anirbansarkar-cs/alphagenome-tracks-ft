"""Read variant-effect result JSONs and report the LEVEL-MATCHED comparison.

Why a third metric is needed: `eval_variant_effect.py` reports an absolute and a relative AUROC
and BOTH are confounded, in opposite directions (see that script's docstring). Neither is
quotable. The fix is level-matched stratification -- peak and background overlap in predicted
level over a real band, so the question can be asked at matched level, where the confound is held
fixed by construction.

It also reports a scale-free CROSS-MODEL test that needs no label: two models trained with
different objectives can predict on very different scales, so their absolute effects are not
comparable, but their RANKINGS over the same positions are. A high Spearman between two models'
effect vectors says they learned the same sequence sensitivity, whatever their calibration.

📒 Intervals bootstrap over LOCI, not positions. Positions inside one window are autocorrelated,
   so a position bootstrap reports an interval several times too narrow.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

RES = Path(os.environ.get("AGTRACKS_RESULTS", "results"))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(u: np.ndarray, v: np.ndarray) -> float:
    ru = np.argsort(np.argsort(u)).astype(np.float64)
    rv = np.argsort(np.argsort(v)).astype(np.float64)
    ru -= ru.mean(); rv -= rv.mean()
    d = float(np.sqrt((ru ** 2).sum() * (rv ** 2).sum()))
    return float((ru * rv).sum() / d) if d > 0 else float("nan")


def load(tag: str, split: str) -> dict | None:
    p = RES / f"ve_{split}_{tag}.json"
    return json.loads(p.read_text()) if p.is_file() and p.stat().st_size else None


def cols(d: dict) -> dict:
    keys = ("label", "local_acc", "local_acc_rel", "local_level_acc", "local_rna",
            "global_acc", "locus", "pos")
    return {k: np.array([r[k] for r in d["rows"]]) for k in keys}


def level_matched_auroc(score, label, level, n_strata=5, min_per=8):
    """AUROC within quantile strata of predicted LEVEL, pooled by stratum size."""
    label = label.astype(bool)
    pk, bg = level[label], level[~label]
    lo, hi = max(pk.min(), bg.min()), min(pk.max(), bg.max())
    band = (level >= lo) & (level <= hi)
    if band.sum() < 2 * min_per:
        return float("nan"), 0, 0, (lo, hi)
    s, l, v = score[band], label[band], level[band]
    edges = np.quantile(v, np.linspace(0, 1, n_strata + 1))
    edges[0] -= 1e-12; edges[-1] += 1e-12
    num = den = 0.0
    used = 0
    for k in range(n_strata):
        m = (v > edges[k]) & (v <= edges[k + 1])
        if m.sum() < min_per or l[m].sum() == 0 or (~l[m]).sum() == 0:
            continue
        w = float(l[m].sum() * (~l[m]).sum())
        num += auroc(s[m], l[m]) * w; den += w; used += 1
    return (num / den if den > 0 else float("nan")), int(band.sum()), used, (lo, hi)


def boot_loci(score, label, level, loci, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(loci)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        m = np.concatenate([np.where(loci == u)[0] for u in pick])
        v, _, used, _ = level_matched_auroc(score[m], label[m], level[m])
        if used > 0 and np.isfinite(v):
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--pair", nargs=2, default=["model_a", "model_b"],
                    metavar=("TAG_A", "TAG_B"))
    ap.add_argument("--replicates", nargs="*", default=[],
                    help="tags to pool as a run-to-run scatter estimate for the same metric")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    print(f"VARIANT EFFECT, split={a.split}. Level-matched AUROC is THE metric; the absolute and")
    print("relative AUROCs are printed only to show the confound they each carry.\n")
    hdr = (f"{'arm':22s} {'n_pos':>6s} {'loci':>5s} {'MATCHED':>16s} {'strata':>7s} "
           f"{'abs':>7s} {'rel':>7s} {'easy':>7s} {'r(abs,lvl)':>11s}")
    print(hdr); print("-" * len(hdr))
    keep = {}
    for tag in list(a.pair) + list(a.replicates):
        d = load(tag, a.split)
        if d is None:
            print(f"{tag:22s} (not scored yet)")
            continue
        c = cols(d)
        keep[tag] = c
        m, nb, used, band = level_matched_auroc(c["local_acc"], c["label"], c["local_level_acc"])
        ci = boot_loci(c["local_acc"], c["label"], c["local_level_acc"], c["locus"], a.boot)
        print(f"{tag:22s} {d['n_positions']:6d} {d['n_loci_used']:5d} "
              f"{m:6.3f} [{ci[0]:.2f},{ci[1]:.2f}] {used:7d} "
              f"{d['auroc_local_acc']:7.3f} {d['auroc_local_acc_rel']:7.3f} "
              f"{d['control_easy_task_level_auroc']:7.3f} "
              f"{d['control_abs_effect_vs_level_spearman']:+11.3f}")
        assert d["control_noop_max_abs_effect"] == 0.0, f"{tag}: no-op control failed"

    ft, at = a.pair
    if ft in keep and at in keep:
        cf, ca = keep[ft], keep[at]
        same = (cf["pos"] == ca["pos"]).all() and (cf["locus"] == ca["locus"]).all()
        print(f"\nCROSS-MODEL, scale-free (same positions: {same}):")
        if same:
            print(f"  Spearman(model A effect, model B effect), all positions : "
                  f"{spearman(cf['local_acc'], ca['local_acc']):+.4f}")
            pk = cf["label"].astype(bool)
            print(f"  Spearman, PEAK positions only                          : "
                  f"{spearman(cf['local_acc'][pk], ca['local_acc'][pk]):+.4f}")
            print(f"  Spearman, BACKGROUND positions only                     : "
                  f"{spearman(cf['local_acc'][~pk], ca['local_acc'][~pk]):+.4f}")
            print("  ⇒ high agreement = the two objectives learned the SAME sequence sensitivity")
            print("    structure, and any difference between them rests on calibration, not sequence.")

    if a.replicates:
        vals = []
        for tag in a.replicates:
            if tag in keep:
                c = keep[tag]
                v, _, u, _ = level_matched_auroc(c["local_acc"], c["label"], c["local_level_acc"])
                if u > 0 and np.isfinite(v):
                    vals.append(v)
        if len(vals) > 1:
            print(f"\nRUN-TO-RUN SCATTER of the level-matched AUROC over {len(vals)} replicates: "
                  f"mean {np.mean(vals):.4f}  sd {np.std(vals, ddof=1):.4f}  "
                  f"range {min(vals):.4f}-{max(vals):.4f}")
            print("  NOTE: a 1-vs-1 comparison on a new metric is not readable")
            print("     until the metric's own run-to-run scatter is known.")


if __name__ == "__main__":
    main()
