#!/usr/bin/env python3
"""Shape and null diagnostics for a trained model, on held-out chromosomes.

WHY THIS EXISTS. Training reported one number: per-track Pearson r within validation windows,
against a mean-profile baseline (0.71 vs 0.166 at the best checkpoint). That is the EASIEST of
the standard metrics and is inflated two ways we have already demonstrated:
  * REPLICATE STRUCTURE -- one study contributes 18 of 48 tracks and carries the headline;
    the other 30 tracks median 0.53.
  * PROFILE SHAPE -- every track peaks at TSSs, so predicting "the average landscape" scores
    well without any sequence-specific knowledge.
This computes the metrics that can actually distinguish learning from those artefacts.

EVERYTHING RUNS ON THE TEST SPLIT (chr9, chr10), which no training or model selection has ever
seen. The validation split (chr8) chose the checkpoint, so reusing it would be mild but real
leakage.

METRIC DECOMPOSITION -- three correlations that answer different questions:
  within_window : flatten positions x windows. What training reported. Easiest.
  across_locus  : per track, correlate WINDOW-MEAN predicted vs observed across windows.
                  Does the model know which REGIONS are active? Immune to profile shape.
  across_track  : at each position, correlate the 48-track profile predicted vs observed.
                  Does the model know which ASSAY/TISSUE is active HERE? Hardest, and the one
                  a profile-memoriser fails outright.

NULLS. A dinucleotide-preserving shuffle keeps composition (GC, CpG) and destroys grammar, so
a model reading regulatory syntax must collapse toward baseline on it. A GC-only predictor
bounds how much of the score is pure composition. Reverse-complement consistency checks the
model did not key on strand artefacts.

Predictions are stored BINNED at 128 bp: 561 test windows x 1,048,576 bp x 48 tracks would be
113 GB at 1 bp, and every metric here is meaningful at 128 bp.

Usage:
    python scripts/eval_profiles.py --checkpoint runs/my_run/last.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agtracks import losses as PL
BIN = 128


def dinuc_shuffle(seq_idx: np.ndarray, rng: random.Random) -> np.ndarray:
    """Altschul-Erikson dinucleotide-preserving shuffle on an integer-coded sequence.

    A mononucleotide shuffle would preserve only GC. Preserving DINUCLEOTIDES is the stricter
    and standard null: it keeps local composition intact so that any drop in performance is
    attributable to destroyed ORDER, not to changed base content.
    """
    n = len(seq_idx)
    if n < 3:
        return seq_idx.copy()
    edges: dict[int, list[int]] = {}
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        edges.setdefault(int(a), []).append(int(b))
    last = int(seq_idx[-1])
    # pick a random last-edge per vertex forming a tree rooted at `last`, then shuffle the rest
    for v in edges:
        rng.shuffle(edges[v])
    out = [int(seq_idx[0])]
    cur = out[0]
    pools = {v: list(e) for v, e in edges.items()}
    for _ in range(n - 1):
        pool = pools.get(cur)
        if not pool:                     # dead end: fall back to any remaining edge
            rem = [v for v, e in pools.items() if e]
            if not rem:
                break
            cur = rem[0]
            pool = pools[cur]
        nxt = pool.pop()
        out.append(nxt)
        cur = nxt
    while len(out) < n:                  # pad with the original tail if the walk ended early
        out.append(int(seq_idx[len(out)]))
    return np.array(out[:n], dtype=seq_idx.dtype)


def pearson(a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def load_tss(gff_gz: Path, chroms: set[str]) -> dict[str, list[tuple[int, str]]]:
    """TSS positions per chromosome from a GFF3. Strand matters: a TSS is the 5' end."""
    out: dict[str, list[tuple[int, str]]] = {c: [] for c in chroms}
    op = gzip.open if str(gff_gz).endswith(".gz") else open
    with op(gff_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[2] != "gene" or f[0] not in chroms:
                continue
            start, end, strand = int(f[3]) - 1, int(f[4]), f[6]
            out[f[0]].append((start if strand == "+" else end - 1, strand))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=None)
    ap.add_argument("--windows", type=Path,
                    default=None)
    ap.add_argument("--manifest", type=Path,
                    default=None)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--gff", type=Path, default=None,
                    help="optional annotation, for the TSS metrics only. Omitted -> those are "
                         "skipped and the rest of the report is unaffected.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-windows", type=int, default=120)
    ap.add_argument("--organism-index", type=int, default=2)
    ap.add_argument("--num-organisms", type=int, default=3)
    ap.add_argument("--skip-nulls", action="store_true")
    ap.add_argument("--rate-param", choices=["direct", "exp", "auto"], default="auto",
                    help="'auto' reads it from the checkpoint's saved args; checkpoints from "
                         "before the fix have no such field and fall back to 'exp'.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from agtracks.dataset import GenomeWindowDataset
    from agtracks.train import build_model

    dev = "cuda"
    ds = GenomeWindowDataset(args.windows, args.manifest, args.fasta,
                              split=args.split, require_local=False)
    T = len(ds.tracks)
    tracks = ds.tracks
    print(f"split={args.split}  windows={len(ds)}  tracks={T}", flush=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Resolve the rate parameterisation from the checkpoint. A pre-fix checkpoint has no
    # 'rate_param' in its saved args and MUST be scored with 'exp', because that is what its
    # weights were fit under; scoring it with 'direct' would be meaningless, not conservative.
    RATE_PARAM = args.rate_param
    if RATE_PARAM == "auto":
        RATE_PARAM = (ck.get("args") or {}).get("rate_param", "exp")
    print(f"  rate parameterisation: {RATE_PARAM}"
          f"{'  (pre-fix checkpoint)' if RATE_PARAM == 'exp' else ''}", flush=True)
    print(f"checkpoint step={ck.get('step')}  metrics={ck.get('metrics', {}).get('r_model_median')}",
          flush=True)
    # build_model refuses a None checkpoint (RMSBatchNorm never updates running_var, so a
    # randomly-initialised model would run with no normalisation and report nothing wrong).
    # Build from the base checkpoint to satisfy that guard, then overwrite with OUR weights.
    model, head = build_model(num_tracks=T, num_organisms=args.num_organisms,
                              checkpoint=None,
                              grad_ckpt=False, device=dev)
    miss_m, unexp_m = model.load_state_dict(ck["model"], strict=False)
    miss_h, unexp_h = head.load_state_dict(ck["head"], strict=False)
    if miss_m or unexp_m or miss_h or unexp_h:
        raise SystemExit(f"checkpoint does not match the model: model missing={len(miss_m)} "
                         f"unexpected={len(unexp_m)}, head missing={len(miss_h)} "
                         f"unexpected={len(unexp_h)}. Evaluating a partially-loaded model "
                         f"would silently measure the BASE checkpoint, not the trained one.")
    model.eval(); head.eval()

    means = np.array([t.get("track_mean") if (t.get("track_mean") or 0) > 0 else 1.0
                      for t in tracks], dtype=np.float32)
    scale = np.clip(means, 1e-3, None)

    step = max(1, len(ds) // args.n_windows)
    idxs = list(range(0, len(ds), step))[:args.n_windows]
    nb = ds.output_bp // BIN
    P = np.zeros((len(idxs), nb, T), dtype=np.float32)   # predicted, 128bp bins
    Y = np.zeros((len(idxs), nb, T), dtype=np.float32)   # observed
    Pn = np.zeros_like(P) if not args.skip_nulls else None   # shuffled-sequence prediction
    Pr = np.zeros_like(P) if not args.skip_nulls else None   # reverse-complement prediction
    GC = np.zeros((len(idxs), nb), dtype=np.float32)
    meta = []
    rng = random.Random(0)
    org = torch.full((1,), args.organism_index, dtype=torch.long, device=dev)

    @torch.no_grad()
    def predict(oh: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(oh).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, org, embeddings_only=True, channels_last=False,
                        resolutions=(1, 128))
            pl = head({1: out["embeddings_1bp"]}, org, return_scaled=True,
                      channels_last=True)[1]
        # MUST match how the checkpoint was trained. Pre-fix runs exponentiated the head
        # output (flooring every prediction at 1.0); fixed runs use it directly as the rate.
        p = PL.rate_from_head(pl.float(), RATE_PARAM).squeeze(0).cpu().numpy()   # (L, T)
        return p.reshape(nb, BIN, T).mean(1)

    for j, i in enumerate(idxs):
        oh, y = ds[i]
        w = ds.windows[i]
        meta.append({"chrom": w["chrom"], "start": w["out_start"]})
        Y[j] = (y / scale).reshape(nb, BIN, T).mean(1)
        P[j] = predict(oh)
        GC[j] = oh[:, 1:3].sum(1).reshape(nb, BIN).mean(1)          # C+G fraction per bin
        if not args.skip_nulls:
            idx_seq = oh.argmax(1).astype(np.int64)
            idx_seq[oh.sum(1) == 0] = 0
            sh = dinuc_shuffle(idx_seq, rng)
            oh_sh = np.zeros_like(oh); oh_sh[np.arange(len(sh)), sh] = 1.0
            Pn[j] = predict(oh_sh)
            oh_rc = oh[::-1, ::-1].copy()                            # A<->T, C<->G and reverse
            Pr[j] = predict(oh_rc)[::-1]
        if (j + 1) % 20 == 0:
            print(f"  [{j+1}/{len(idxs)}] windows done", flush=True)

    # rate_param is recorded because it is NOT inferable from the report otherwise: a pre-fix
    # checkpoint carries no rate_param in its saved args and `auto` silently resolves it to exp.
    # A number scored under the wrong parameterisation is a number for a different model, and
    # without this field there is no way to tell after the fact which one produced it.
    rep: dict = {"split": args.split, "n_windows": len(idxs), "n_tracks": T,
                 "rate_param": RATE_PARAM, "checkpoint": str(args.checkpoint),
                 "bin_bp": BIN}

    # ---- baselines -------------------------------------------------------------------
    meanprof = Y.mean(0, keepdims=True).repeat(len(idxs), 0)

    def within(A):
        return pearson(A.reshape(-1, T), Y.reshape(-1, T), axis=0)

    r_model = within(P)
    r_base = within(meanprof)

    # LEAVE-ONE-OUT is the honest control. `meanprof` above averages ALL windows INCLUDING the
    # one it predicts, so each window supplies 1/N of its own answer and the control leaks --
    # and the leak scales with 1/N. Measured on chr9+10: 0.153 at N=40, 0.087 at N=120, i.e.
    # the "baseline" mostly reports the sample size. A deployed mean-profile predictor would be
    # fit on OTHER data, so building the profile from the other windows is what it should have
    # been. LOO lands at ~-0.009 -- a mean profile of other loci says nothing about this one.
    # Both are reported; quote the LOO one, and never quote a model/baseline RATIO.
    meanprof_loo = (Y.sum(0, keepdims=True) - Y) / (len(idxs) - 1)
    r_base_loo = within(meanprof_loo)

    rep["within_window"] = {"model_median": float(np.median(r_model)),
                            "meanprofile_median": float(np.median(r_base_loo)),
                            "meanprofile_median_selfincluded": float(np.median(r_base)),
                            "n_windows_in_meanprofile": len(idxs),
                            "beats_baseline": int((r_model > r_base_loo).sum()),
                            "beats_baseline_selfincluded": int((r_model > r_base).sum())}

    # across-locus: does it know WHICH REGIONS are active? (window means, profile-shape immune)
    r_loc = pearson(P.mean(1), Y.mean(1), axis=0)
    rep["across_locus"] = {"median": float(np.median(r_loc)),
                           "q25": float(np.percentile(r_loc, 25)),
                           "q75": float(np.percentile(r_loc, 75))}

    # across-track: at each position, is the 48-track PROFILE right? (specificity)
    Pf = P.reshape(-1, T); Yf = Y.reshape(-1, T)
    keep = Yf.std(1) > 1e-6
    r_trk = pearson(Pf[keep].T, Yf[keep].T, axis=0)
    rep["across_track"] = {"median": float(np.median(r_trk)),
                           "n_positions": int(keep.sum())}

    # ---- nulls -----------------------------------------------------------------------
    if not args.skip_nulls:
        r_shuf = within(Pn)
        r_rc = pearson(Pr.reshape(-1, T), P.reshape(-1, T), axis=0)
        gc_pred = np.repeat(GC[:, :, None], T, axis=2)
        r_gc = within(gc_pred)
        rep["nulls"] = {
            "shuffled_median": float(np.median(r_shuf)),
            "shuffled_drop": float(np.median(r_model) - np.median(r_shuf)),
            "gc_only_median": float(np.median(r_gc)),
            "rc_consistency_median": float(np.median(r_rc)),
        }

    # ---- per-assay and per-study, because the aggregate is replicate-inflated ---------
    from collections import defaultdict
    for key in ("assay", "study"):
        g = defaultdict(list)
        for t, v, vl, in zip(tracks, r_model, r_loc):
            g[t.get(key)].append((float(v), float(vl)))
        rep[f"by_{key}"] = {str(k): {"n": len(v),
                                     "within_median": float(np.median([a for a, _ in v])),
                                     "across_locus_median": float(np.median([b for _, b in v]))}
                            for k, v in sorted(g.items())}

    # ---- tissue specificity: 22 RNA tracks are 22 DISTINCT TISSUES -------------------
    rna = [i for i, t in enumerate(tracks) if t["assay"] == "RNA-Seq"]
    if len(rna) >= 3:
        Pr_ = P.reshape(-1, T)[:, rna]; Yr_ = Y.reshape(-1, T)[:, rna]
        k2 = Yr_.std(1) > 1e-6
        r_tis = pearson(Pr_[k2].T, Yr_[k2].T, axis=0)
        rep["tissue_specificity_rna"] = {"n_tissues": len(rna),
                                         "median_across_tissue_r": float(np.median(r_tis)),
                                         "n_positions": int(k2.sum())}

    # ---- peak recovery: top-1% observed bins per track, ranked by prediction ---------
    aps = []
    for t in range(T):
        y = Y[:, :, t].ravel(); p = P[:, :, t].ravel()
        thr = np.quantile(y, 0.99)
        pos = y >= thr
        if pos.sum() < 10:
            continue
        order = np.argsort(-p)
        tp = np.cumsum(pos[order])
        prec = tp / np.arange(1, len(order) + 1)
        aps.append(float((prec * pos[order]).sum() / pos.sum()))
    if aps:
        rep["peak_ap_top1pct"] = {"median": float(np.median(aps)), "n_tracks": len(aps),
                                  "random_baseline": 0.01}

    # ---- TSS enrichment --------------------------------------------------------------
    if args.gff is not None and args.gff.exists():
        chroms = {m["chrom"] for m in meta}
        tss = load_tss(args.gff, chroms)
        half = 16                                    # +/- 16 bins = +/- 2,048 bp
        accP = np.zeros(2 * half); accY = np.zeros(2 * half); nT = 0
        for j, m in enumerate(meta):
            lo, hi = m["start"], m["start"] + ds.output_bp
            for pos, _s in tss.get(m["chrom"], []):
                if lo + half * BIN <= pos < hi - half * BIN:
                    b = (pos - lo) // BIN
                    accP += P[j, b - half:b + half].mean(1)
                    accY += Y[j, b - half:b + half].mean(1)
                    nT += 1
        if nT:
            fp, fy = accP / nT, accY / nT
            rep["tss"] = {"n_tss": nT,
                          "pred_enrichment": float(fp[half - 2:half + 2].mean() /
                                                   max(fp[:4].mean(), 1e-9)),
                          "obs_enrichment": float(fy[half - 2:half + 2].mean() /
                                                  max(fy[:4].mean(), 1e-9)),
                          "profile_r": float(pearson(fp[:, None], fy[:, None])[0])}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    np.savez_compressed(args.out.with_suffix(".preds.npz"), P=P, Y=Y, GC=GC,
                        track_ids=np.array([t["track_id"] for t in tracks]))
    print("\n" + json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    print("EVAL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
