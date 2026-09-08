"""In-silico single-nucleotide variant effects, with the confounds controlled.

Substitute one base, re-run the model, and measure the predicted change. Then ask whether
substitutions inside observed peaks produce larger effects than substitutions in background --
a label-free, rank-based discrimination task that needs no variant truth set.

🚨 THE TWO OBVIOUS READOUTS ARE BOTH TAUTOLOGICAL, IN OPPOSITE DIRECTIONS. This is the whole
   reason the script is more than ten lines, and it is easy to get wrong:
     absolute |Delta|        scales with how large the prediction already is, so it ranks almost
                             identically to the predicted LEVEL. Measured: AUROC 0.94-0.97 with a
                             rank correlation of +0.90 against level, while level ALONE scores
                             0.98. It restates "predictions are bigger in peaks".
     relative |Delta|/level  inverts, because a near-zero background denominator inflates the
                             ratio. Measured: 0.13-0.44, anti-correlated with level.
   ⇒ **Neither may be quoted.** The headline is the LEVEL-MATCHED AUROC: stratify positions into
     quantiles of predicted level and compute the AUROC within strata, so the confound is held
     fixed by construction. All three are reported side by side so the confound stays visible.

🚨 FOUR CONTROLS, NONE OPTIONAL:
   1. NO-OP SUBSTITUTION. Replacing the reference base with itself must give EXACTLY zero. This is
      the guard against the dominant failure: if the alternate sequence is not actually different
      -- a bad index, an aliased copy, an all-zero one-hot row -- every effect is 0, the AUROC
      lands at 0.500, and that reads as a clean negative rather than a broken script. Asserted.
   2. LABEL PERMUTATION NULL. Shuffled labels must give ~0.5.
   3. TRIVIAL SEQUENCE BASELINE. AUROC from |GC change| alone. The model must beat it.
   4. THE EASY VERSION FIRST. AUROC using the prediction LEVEL, no variant at all. If the model
      cannot separate peak from background by level, its variant numbers mean nothing.

📒 Positions inside one window are autocorrelated, so `--n-loci` is the unit of independence and
   every interval bootstraps over LOCI, not positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ACC_ASSAYS = ("ATAC-seq", "DNase-Hypersensitivity")
BASES = "ACGT"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie correction. No sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks within ties, so a score vector of all-identical values gives exactly 0.5
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt-name", default="last.pt",
                    help="last.pt, never best.pt (best.pt is selected on within-window r)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--n-loci", type=int, default=8,
                    help="windows sampled; loci are the unit of INDEPENDENCE")
    ap.add_argument("--pos-per-locus", type=int, default=48,
                    help="variant positions per window, half peak and half background")
    ap.add_argument("--loci-seed", type=int, default=0)
    ap.add_argument("--local-bp", type=int, default=128,
                    help="half-width of the local effect window around the variant")
    ap.add_argument("--peak-q", type=float, default=0.98,
                    help="observed-accessibility quantile at or above which a position is a PEAK")
    ap.add_argument("--bg-q", type=float, default=0.50,
                    help="quantile at or below which a position is BACKGROUND")
    ap.add_argument("--edge-margin", type=int, default=4096,
                    help="output positions this close to either edge are never chosen")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from agtracks.dataset import GenomeWindowDataset
    from agtracks import losses as PL
    from agtracks.train import build_model

    ck = torch.load(a.run / a.ckpt_name, map_location="cpu", weights_only=False)
    t = ck["args"]
    if t.get("target_clip", "none") != "none":
        raise SystemExit("assumes --target-clip none (as every scored arm is)")
    bs = int(t.get("bin_size", 1))
    if bs != 1:
        raise SystemExit(f"single-base substitution needs bin_size 1, run used {bs}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = GenomeWindowDataset(Path(t["windows"]), Path(t["manifest"]), Path(t["fasta"]),
                              split=a.split, bin_size=bs,
                              max_tracks=t.get("max_tracks"), track_seed=t.get("track_seed", 0))
    T = ds.n_tracks()
    assay = [str(x.get("assay", "?")) for x in ds.tracks]
    acc = np.array([i for i, s in enumerate(assay) if s in ACC_ASSAYS], dtype=np.int64)
    rna = np.array([i for i, s in enumerate(assay) if s not in ACC_ASSAYS], dtype=np.int64)
    if acc.size == 0:
        raise SystemExit("no accessibility tracks; the peak/background label is undefined")
    _raw = [x.get("track_mean") for x in ds.tracks]
    means = np.clip(np.array([m if (m is not None and m > 0) else 1.0 for m in _raw],
                             dtype=np.float32), 1e-3, None)
    scale = torch.tensor(1.0 / means, device=dev)

    model, head = build_model(T, t["num_organisms"], Path(t["checkpoint"]), False, dev)
    model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    crop, out_bp = ds.crop, ds.output_bp
    n_bins, crop_bins = out_bp // bs, crop // bs
    acc_g = torch.tensor(acc, device=dev)
    rna_g = torch.tensor(rna, device=dev)

    rng = np.random.default_rng(a.loci_seed)
    n_avail = len(ds)
    idx = (np.arange(n_avail) if n_avail <= a.n_loci
           else np.sort(rng.choice(n_avail, a.n_loci, replace=False)))

    org = torch.full((1,), t["organism_index"], dtype=torch.long, device=dev)

    def predict(x_oh: torch.Tensor) -> torch.Tensor:
        """(1,L,4) one-hot -> (n_bins, T) rate predictions, cropped to the output window."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            emb = model(x_oh, org, embeddings_only=True, channels_last=False, resolutions=(bs,))
            key = "embeddings_1bp" if bs == 1 else "embeddings_128bp"
            P = head({bs: emb[key]}, org, return_scaled=True, channels_last=True)[bs]
        if P.shape[1] != n_bins:
            if P.shape[1] != n_bins + 2 * crop_bins:
                raise RuntimeError(f"pred len {P.shape[1]} unexpected")
            P = P[:, crop_bins:crop_bins + n_bins, :]
        return PL.rate_from_head(P.float(), t.get("rate_param", "direct"))[0]

    rows = []          # one dict per (locus, position)
    noop_max = 0.0     # control 1: must stay exactly 0
    n_noop = 0

    for c, w in enumerate(idx):
        oh_np, y_np = ds[int(w)]
        y_acc = y_np[:, acc].mean(axis=1)                       # observed accessibility profile
        x_ref = torch.from_numpy(oh_np)[None].to(dev)
        P_ref = predict(x_ref)                                  # (n_bins, T)
        ref_acc_level = P_ref[:, acc_g].mean(1)                 # for the EASY-task control

        lo, hi = a.edge_margin, n_bins - a.edge_margin
        if hi <= lo:
            raise SystemExit("edge margin leaves no positions")
        band = np.arange(lo, hi)
        # A position is only usable if its reference base is a definite A/C/G/T: the one-hot
        # encodes N as an all-zero row, and substituting into an all-zero row would compare two
        # different KINDS of sequence rather than two alleles.
        known = oh_np[crop + band].sum(axis=1) == 1.0
        band = band[known]
        if band.size < a.pos_per_locus:
            print(f"  locus {int(w)}: only {band.size} usable positions, skipping")
            continue
        v = y_acc[band]
        thr_hi = np.quantile(y_acc[lo:hi], a.peak_q)
        thr_lo = np.quantile(y_acc[lo:hi], a.bg_q)
        pk, bg = band[v >= thr_hi], band[v <= thr_lo]
        k = a.pos_per_locus // 2
        if pk.size < k or bg.size < k:
            print(f"  locus {int(w)}: peak {pk.size} bg {bg.size} < {k}, skipping")
            continue
        chosen = np.concatenate([rng.choice(pk, k, replace=False),
                                 rng.choice(bg, k, replace=False)])
        labels = np.concatenate([np.ones(k, bool), np.zeros(k, bool)])

        for pos, lab in zip(chosen, labels):
            ip = crop + int(pos)
            ref_b = int(np.argmax(oh_np[ip]))
            s0, s1 = max(0, int(pos) - a.local_bp), min(n_bins, int(pos) + a.local_bp + 1)
            lvl = float(P_ref[s0:s1][:, acc_g].mean())   # local predicted level, for the ratio
            loc_acc, loc_rna, glo_acc, dgc = [], [], [], []
            for alt_b in range(4):
                x = x_ref.clone()
                x[0, ip, :] = 0.0
                x[0, ip, alt_b] = 1.0
                P = predict(x)
                d = (P - P_ref).abs()
                if alt_b == ref_b:
                    # CONTROL 1: a no-op substitution must change nothing, exactly.
                    noop_max = max(noop_max, float(d.max())); n_noop += 1
                    continue
                loc_acc.append(float(d[s0:s1][:, acc_g].mean()))
                loc_rna.append(float(d[s0:s1][:, rna_g].mean()))
                glo_acc.append(float(d[:, acc_g].mean()))
                dgc.append(abs(int(BASES[alt_b] in "GC") - int(BASES[ref_b] in "GC")))
            ma = float(np.mean(loc_acc))
            rows.append({"locus": int(w), "pos": int(pos), "label": bool(lab),
                         "local_acc": ma,
                         # RELATIVE effect: divides out the local level, so a win here is
                         # sequence sensitivity and not a restatement of the easy task.
                         "local_acc_rel": ma / (lvl + 1e-6),
                         "local_level_acc": lvl,
                         "local_rna": float(np.mean(loc_rna)),
                         "global_acc": float(np.mean(glo_acc)),
                         "gc_change": float(np.mean(dgc)),
                         "ref_level_acc": float(ref_acc_level[int(pos)])})
        print(f"  locus {int(w)} ({c + 1}/{len(idx)}): {len(rows)} positions scored", flush=True)

    if not rows:
        raise SystemExit("FATAL: zero positions scored -- refusing to write an empty result")

    lab = np.array([r["label"] for r in rows])
    got = {k: np.array([r[k] for r in rows]) for k in
           ("local_acc", "local_acc_rel", "local_level_acc", "local_rna", "global_acc",
            "gc_change", "ref_level_acc")}

    def spearman(u, v):
        ru = np.argsort(np.argsort(u)).astype(np.float64)
        rv = np.argsort(np.argsort(v)).astype(np.float64)
        ru -= ru.mean(); rv -= rv.mean()
        d = float(np.sqrt((ru ** 2).sum() * (rv ** 2).sum()))
        return float((ru * rv).sum() / d) if d > 0 else float("nan")

    # CONTROL 2: the matched null for the metric -- labels shuffled, 32 draws.
    perm = [auroc(got["local_acc_rel"], rng.permutation(lab)) for _ in range(32)]

    res = {
        "run": a.run.name, "ckpt": a.ckpt_name, "split": a.split,
        "loss": t["loss"], "gene_tissue_weight": t["gene_tissue_weight"],
        "poisson_weight": t["poisson_weight"], "lr": t["lr"], "seed": t["seed"],
        "step": ck.get("step"), "manifest": Path(t["manifest"]).name,
        "n_tracks": T, "n_acc_tracks": int(acc.size), "n_rna_tracks": int(rna.size),
        "n_loci_requested": int(a.n_loci), "n_loci_used": len({r["locus"] for r in rows}),
        "n_positions": len(rows), "n_peak": int(lab.sum()), "n_background": int((~lab).sum()),
        "local_bp": a.local_bp, "peak_q": a.peak_q, "bg_q": a.bg_q,
        # ---- THE HEADLINE: relative effect, level divided out ----
        "auroc_local_acc_rel": auroc(got["local_acc_rel"], lab),
        # ---- the absolute effect: real, but partly a restatement of the level. Never quote alone.
        "auroc_local_acc": auroc(got["local_acc"], lab),
        "auroc_local_rna": auroc(got["local_rna"], lab),
        "auroc_global_acc": auroc(got["global_acc"], lab),
        # ---- the four controls ----
        "control_noop_max_abs_effect": noop_max,
        "control_noop_n": n_noop,
        "control_perm_null_mean": float(np.mean(perm)),
        "control_perm_null_sd": float(np.std(perm, ddof=1)),
        "control_gc_baseline_auroc": auroc(got["gc_change"], lab),
        "control_easy_task_level_auroc": auroc(got["ref_level_acc"], lab),
        # CONTROL 5: how much of the absolute effect IS the level? A rank correlation near 1 means
        # the absolute AUROC carries no information beyond control 4.
        "control_abs_effect_vs_level_spearman": spearman(got["local_acc"], got["local_level_acc"]),
        "control_rel_effect_vs_level_spearman": spearman(got["local_acc_rel"],
                                                         got["local_level_acc"]),
        # ---- effect magnitudes, so a tie in AUROC can be read against the scale ----
        "effect_local_acc_median_peak": float(np.median(got["local_acc"][lab])),
        "effect_local_acc_median_bg": float(np.median(got["local_acc"][~lab])),
        "effect_global_acc_median": float(np.median(got["global_acc"])),
        "rows": rows,
    }

    # 🚨 GUARD THE OUTPUT, NOT THE EXIT CODE.
    assert res["n_positions"] > 0
    assert res["n_peak"] > 0 and res["n_background"] > 0, "one class empty; AUROC undefined"
    # Control 1 is a hard gate: a no-op substitution that moves the prediction means the
    # substitution path is wrong, and every effect below is suspect.
    assert n_noop > 0, "the no-op control never ran; substitution path unverified"
    assert noop_max == 0.0, (f"NO-OP SUBSTITUTION CHANGED THE PREDICTION by {noop_max:.3e} -- the "
                             "substitution path is broken; refusing to report variant effects")
    if not (0.40 <= res["control_perm_null_mean"] <= 0.60):
        raise SystemExit(f"permutation null is {res['control_perm_null_mean']:.3f}, not ~0.5; "
                         "the AUROC implementation or the labels are wrong")

    print(f"\n{res['run']} {a.split} step={res['step']} loss={res['loss']} lr={res['lr']}")
    print(f"  positions {res['n_positions']} ({res['n_peak']} peak / {res['n_background']} bg) "
          f"over {res['n_loci_used']} loci")
    print(f"  AUROC RELATIVE effect (headline) local ACC {res['auroc_local_acc_rel']:.4f}")
    print(f"  AUROC absolute effect            local ACC {res['auroc_local_acc']:.4f}   "
          f"local RNA {res['auroc_local_rna']:.4f}   global ACC {res['auroc_global_acc']:.4f}")
    print(f"  CONTROLS  no-op {noop_max:.3e} (n={n_noop})   "
          f"perm null {res['control_perm_null_mean']:.4f}+-{res['control_perm_null_sd']:.4f}   "
          f"GC-only {res['control_gc_baseline_auroc']:.4f}   "
          f"EASY (level) {res['control_easy_task_level_auroc']:.4f}")
    print(f"  effect median  peak {res['effect_local_acc_median_peak']:.5f}  "
          f"bg {res['effect_local_acc_median_bg']:.5f}")
    print(f"  TAUTOLOGY GUARD  rank corr(|effect|, level) = "
          f"{res['control_abs_effect_vs_level_spearman']:+.4f}   "
          f"corr(relative effect, level) = "
          f"{res['control_rel_effect_vs_level_spearman']:+.4f}")
    if abs(res["control_abs_effect_vs_level_spearman"]) > 0.90:
        print("  ⚠️ the ABSOLUTE effect is ~a monotone function of the predicted level, so its "
              "AUROC restates control 4. Quote the RELATIVE number.")
    if res["control_easy_task_level_auroc"] < 0.60:
        print("  ⚠️ THE EASY TASK IS NEAR CHANCE. Do not report the variant number: a model that "
              "cannot separate peak from background by prediction LEVEL has not earned a reading "
              "of its variant effects.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
