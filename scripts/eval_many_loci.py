#!/usr/bin/env python3
"""Re-score a trained run on MANY held-out loci, at its NATIVE geometry.

WHY. Every history.json number in this project was computed on 8 FIXED windows: the val loader
is `shuffle=False` and validate() breaks at --val-batches, so at batch-size 1 it scores the first
8 windows of the split, the same 8 every time (`n_positions_scored = 16384 = 8 x 2048` says so).
Averaging 20 validations does NOT widen that -- it averages training noise over the same 8 loci.
For the 1 Mb runs that is 8 of 293 windows, 2.7% of the split.

Arms sharing a window index at least share those 8 loci, so their RANKING is internally valid --
but a seed sd measured at fixed loci carries NO LOCUS-SAMPLING VARIANCE, which is why stage 2's
"E5 beats E3 by 7.1x seed sd" cannot be taken at face value. This script supplies the missing
component by scoring many loci.

DIFFERENT FROM w1_common_eval.py ON PURPOSE. That script crops every arm to a common 2,048 bp
window so arms trained at DIFFERENT geometries can be compared. Here the arms already share a
window index, so nothing needs cropping and the honest fix is simply MORE LOCI at the geometry
the model was trained for. Cropping to 2 kb would silently change the task.

🚨 LOCI, NOT POSITIONS, ARE THE UNIT OF EVIDENCE. Positions inside a window are ~1 kb
autocorrelated, so a big `n_positions` from few windows is not independent support -- that is
exactly what made 16,384 look reassuring. This script reports `n_loci` first and subsamples
positions PER WINDOW so the position budget is spread across loci rather than concentrated.

Statistics stream: a 1 Mb window is (1,048,576 x T) floats, so concatenating even 64 of them
would be ~16 GB. Per-track r uses pooled sufficient statistics in float64; across-track r keeps a
per-window position subsample.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt-name", default="last.pt")
    ap.add_argument("--split", default="val",
                    help="val = chr8 (what history.json reports, so directly comparable); "
                         "test = chr9/10, never tuned on")
    ap.add_argument("--n-loci", type=int, default=96,
                    help="distinct windows to score. THE headline support number.")
    ap.add_argument("--pos-per-locus", type=int, default=6000,
                    help="live positions sampled per window for across-track r, so the budget is "
                         "spread over loci instead of concentrated in a few")
    ap.add_argument("--loci-seed", type=int, default=0,
                    help="FIXED across arms -- varying it per arm would reintroduce the very "
                         "confound this script removes")
    ap.add_argument("--track-subset", type=int, default=0,
                    help="score only the FIRST N tracks. THE COMMON-BENCHMARK FLAG: "
                         "a narrower manifest must be a strict PREFIX of the wider one -- verify that "
                         "manifest_62 (VERIFIED by path equality, indices 0..47 in order), so "
                         "--track-subset 48 puts 48-track F3 runs and 62-track ladder runs on "
                         "one identical target set. 0 = all tracks.")
    ap.add_argument("--assay-strata", choices=("binary","full"), default="binary",
                    help="binary = ACC (DNase+ATAC) vs RNA, the historical grouping that every\n"
                         "existing the common axis number uses. full = one stratum per assay, so DNase and\n"
                         "ATAC are reported separately. ⚠️ ATAC has only 3 tracks in the 48-track\n"
                         "benchmark, and across-track r over 3 tracks is a very noisy statistic --\n"
                         "that is WHY the historical grouping merges them, not an oversight.")
    ap.add_argument("--pool-bp", type=int, default=1,
                    help="average BOTH prediction and target down the POSITION axis by this "
                         "factor before any statistic. 1 = native, the default, so every "
                         "existing the common axis number is reproduced byte-for-byte. 128 exists to "
                         "meet E0 on ITS resolution: AlphaGenome's accessibility heads are "
                         "128 bp-native, so a 1 bp trained-arm score and a 128 bp zero-shot "
                         "score are NOT comparable -- pooling removes fine-scale noise and "
                         "inflates r, which flatters whichever side is pooled. Pool both or "
                         "compare neither.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from agtracks.dataset import GenomeWindowDataset
    from agtracks import losses as PL
    from agtracks.train import build_model

    ck = torch.load(a.run / a.ckpt_name, map_location="cpu", weights_only=False)
    t = ck["args"]
    if t.get("target_clip", "none") != "none":
        raise SystemExit("assumes --target-clip none")
    bs = int(t.get("bin_size", 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = GenomeWindowDataset(Path(t["windows"]), Path(t["manifest"]), Path(t["fasta"]),
                              split=a.split, bin_size=bs,
                              max_tracks=t.get("max_tracks"), track_seed=t.get("track_seed", 0))
    T = ds.n_tracks()
    # Teff is what we SCORE; T is what the model EMITS. The model always runs at full width --
    # subsetting after the forward pass, never before, so a 62-track model is evaluated exactly
    # as it was trained and only the comparison is narrowed.
    TS = int(a.track_subset)
    if TS:
        if TS > T:
            raise SystemExit(f"--track-subset {TS} > this run's {T} tracks")
        if False:   # set a manifest allow-list here if you narrow across two manifests
            raise SystemExit(f"--track-subset assumes the 48/62 nesting; this run used "
                             f"{Path(t['manifest']).name}. Verify the prefix property first.")
    Teff = TS or T
    _raw = [x.get("track_mean") for x in ds.tracks]
    means = np.clip(np.array([m if (m is not None and m > 0) else 1.0 for m in _raw],
                             dtype=np.float32), 1e-3, None)
    scale = torch.tensor(1.0 / means, device=dev)

    rng = np.random.default_rng(a.loci_seed)
    n_avail = len(ds)
    idx = (np.arange(n_avail) if n_avail <= a.n_loci
           else np.sort(rng.choice(n_avail, a.n_loci, replace=False)))

    model, head = build_model(T, t["num_organisms"], Path(t["checkpoint"]), False, dev)
    model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    crop, out_bp = ds.crop, ds.output_bp
    n_bins, crop_bins = out_bp // bs, crop // bs

    # pooled per-track sufficient statistics, float64
    n = 0
    sx = np.zeros(Teff, np.float64); sy = np.zeros(Teff, np.float64)
    sxx = np.zeros(Teff, np.float64); syy = np.zeros(Teff, np.float64)
    sxy = np.zeros(Teff, np.float64)
    # per-track coverage, for the granular loss comparison (PI request 2026-08-28)
    tP = np.zeros(Teff, np.float64); tY = np.zeros(Teff, np.float64)
    acr, sumP, sumY, n_live_tot, n_pos_tot = [], 0.0, 0.0, 0, 0
    # 🚨 ASSAY-STRATIFIED ACROSS-TRACK r. The pooled number on a mixed RNA+accessibility panel is
    # dominated by the BETWEEN-ASSAY level contrast, and that contrast is CHROMOSOME-DEPENDENT:
    # `track_mean` normalisation is genome-wide, but RNA sits at 2.49x its genome-wide mean on
    # chr9/10 vs 1.22x on chr8 while accessibility barely moves (1.44 vs 1.32). Measured on the
    # constant-per-track baseline, pooled across-track r therefore FLIPS SIGN between splits
    # (+0.534 val, -0.244 test) while each assay group stays stable. So the pooled metric partly
    # measures "did you get the RNA-vs-accessibility offset right on this chromosome", NOT tissue
    # specificity -- which is the WITHIN-assay, across-tissue quantity you usually want.
    _assay = [str(x.get("assay", "?")) for x in ds.tracks[:Teff]]
    _grp = (_assay if a.assay_strata == "full"
            else ["RNA" if x.upper().find("RNA") >= 0 else "ACC" for x in _assay])
    strata = {g: torch.tensor([i for i, gg in enumerate(_grp) if gg == g], device=dev)
              for g in sorted(set(_grp))}
    acr_s = {g: [] for g in strata}

    gen = torch.Generator().manual_seed(a.loci_seed)
    with torch.no_grad():
        for c, w in enumerate(idx):
            oh, y = ds[int(w)]
            x = torch.from_numpy(oh)[None].to(dev)
            Y = (torch.from_numpy(y)[None].to(dev) * scale).float()
            org = torch.full((1,), t["organism_index"], dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model(x, org, embeddings_only=True, channels_last=False, resolutions=(bs,))
                key = "embeddings_1bp" if bs == 1 else "embeddings_128bp"
                P = head({bs: emb[key]}, org, return_scaled=True, channels_last=True)[bs]
            if P.shape[1] != n_bins:
                if P.shape[1] != n_bins + 2 * crop_bins:
                    raise RuntimeError(f"pred len {P.shape[1]} unexpected")
                P = P[:, crop_bins:crop_bins + n_bins, :]
            P = PL.rate_from_head(P.float(), t.get("rate_param", "direct"))
            # 🚨 SUBSET BEFORE THE LIVE FILTER, not after. `live` is defined by variation ACROSS
            # tracks, so computing it on 62 and then narrowing to 48 would select a different
            # position set than a native 48-track run does -- silently un-matching the benchmark.
            if TS:
                P = P[:, :, :TS]; Y = Y[:, :, :TS]
            # 🚨 POOL BEFORE THE LIVE FILTER AND BEFORE EVERY STATISTIC. `live` is
            # Y.std(tracks) > 1e-6 evaluated per position, so pooling afterwards would
            # select positions at one resolution and score them at another.
            if a.pool_bp > 1:
                B, L0, Tn = P.shape
                L = (L0 // a.pool_bp) * a.pool_bp
                P = P[:, :L, :].reshape(B, L // a.pool_bp, a.pool_bp, Tn).mean(2)
                Y = Y[:, :L, :].reshape(B, L // a.pool_bp, a.pool_bp, Tn).mean(2)
            Pf = P[0].double(); Yf = Y[0].double()          # (L, T)
            n += Pf.shape[0]
            sx += Pf.sum(0).cpu().numpy(); sy += Yf.sum(0).cpu().numpy()
            sxx += (Pf * Pf).sum(0).cpu().numpy(); syy += (Yf * Yf).sum(0).cpu().numpy()
            sxy += (Pf * Yf).sum(0).cpu().numpy()
            sumP += float(P.sum()); sumY += float(Y.sum())
            tP += Pf.sum(0).cpu().numpy(); tY += Yf.sum(0).cpu().numpy()
            live = Y[0].std(1) > 1e-6
            Pl, Yl = P[0][live], Y[0][live]
            n_live_tot += int(live.sum()); n_pos_tot += int(live.numel())
            if Pl.shape[0]:
                if Pl.shape[0] > a.pos_per_locus:
                    s = torch.randperm(Pl.shape[0], generator=gen)[:a.pos_per_locus].to(dev)
                    Pl, Yl = Pl[s], Yl[s]
                da = Pl - Pl.mean(1, keepdim=True); db = Yl - Yl.mean(1, keepdim=True)
                acr.append(((da * db).sum(1) /
                            (da.norm(dim=1) * db.norm(dim=1)).clamp_min(1e-8)).cpu())
            # Each stratum gets its OWN live filter and its OWN position subsample, because
            # "varies across tracks" is only meaningful relative to the track set being scored:
            # a position where only RNA varies is live for the pooled metric and dead for ACC.
            for g, cols in strata.items():
                if cols.numel() < 2:
                    continue
                Pg, Yg = P[0][:, cols], Y[0][:, cols]
                lg = Yg.std(1) > 1e-6
                Pg, Yg = Pg[lg], Yg[lg]
                if not Pg.shape[0]:
                    continue
                if Pg.shape[0] > a.pos_per_locus:
                    sg = torch.randperm(Pg.shape[0], generator=gen)[:a.pos_per_locus].to(dev)
                    Pg, Yg = Pg[sg], Yg[sg]
                dc = Pg - Pg.mean(1, keepdim=True); dd = Yg - Yg.mean(1, keepdim=True)
                acr_s[g].append(((dc * dd).sum(1) /
                                 (dc.norm(dim=1) * dd.norm(dim=1)).clamp_min(1e-8)).cpu())
            if (c + 1) % 16 == 0:
                print(f"  {c+1}/{len(idx)} loci", flush=True)

    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx**2, 0) * np.maximum(n * syy - sy**2, 0))
    r_model = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    A = torch.cat(acr) if acr else torch.zeros(1)

    res = {"run": a.run.name, "split": a.split, "step": ck.get("step"),
           "n_loci": int(len(idx)), "n_loci_available": int(n_avail),
           "pct_of_split": round(100 * len(idx) / max(n_avail, 1), 2),
           "n_positions_per_locus": int(out_bp // bs // a.pool_bp),
           "pool_bp": int(a.pool_bp),
           "n_across_samples": int(A.numel()),
           "n_tracks": Teff, "n_tracks_model": T, "track_subset": TS,
           "assay_strata": a.assay_strata,
           "manifest": Path(t["manifest"]).name,
           "n_positions_live": n_live_tot, "n_positions_total": n_pos_tot,
           "output_bp": out_bp, "input_bp": ds.input_bp,
           "r_across_median": float(A.median()),
           "r_across_iqr": [float(A.quantile(.25)), float(A.quantile(.75))],
           "r_model_median": float(np.median(r_model)),
           "cov_ratio": sumP / max(sumY, 1e-6),
           # ---- per-track detail: the granular view the medians hide ----
           "r_across_by_assay": {
               g: (lambda A_: {"median": float(A_.median()), "n_tracks": int(strata[g].numel()),
                               "n_samples": int(A_.numel()),
                               "iqr": [float(A_.quantile(.25)), float(A_.quantile(.75))]})(
                   torch.cat(v)) for g, v in acr_s.items() if v},
           "r_model_per_track": [float(x) for x in r_model],
           "cov_ratio_per_track": [float(p / q) if q > 0 else None for p, q in zip(tP, tY)],
           "track_labels": [{"i": i, "assay": x.get("assay"), "tissue": x.get("tissue"),
                             "study": x.get("study"), "track_id": x.get("track_id")}
                            for i, x in enumerate(ds.tracks[:Teff])]}
    out = a.out or (None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
