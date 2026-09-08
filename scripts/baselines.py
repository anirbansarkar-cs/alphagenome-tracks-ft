#!/usr/bin/env python3
"""SEQUENCE-FREE BASELINES -- the floors, on the SAME axis as every trained arm.

WHY THIS EXISTS (PI request, 2026-08-28: "add zero-shot and mean-signal baseline to
comparisons"). Every the common axis number is a correlation, and a correlation is only interpretable
against what a model that learned NOTHING would score. Two floors, both model-free:

  meanprofile_loo   at each position and track, the average signal over the OTHER 95 loci.
                    LEAVE-ONE-OUT is not a detail: averaging all 96 including the one being
                    predicted leaks 1/N of the answer, and the leak scales with 1/N -- that
                    exact bug made a previous control read 0.153 at N=40 and 0.087 at N=120
                    on the SAME data, i.e. it was reporting the sample size, not a baseline.
  trackmean_loo     one constant per track (its mean over the other 95 loci, all positions).
                    NO positional information whatsoever.

🥇 WHY THE SECOND ONE IS THE IMPORTANT ONE. On WITHIN-WINDOW r a constant prediction has zero
variance, so trackmean scores exactly 0 and meanprofile scores ~0.003 -- both floors are
uninformative there, which is why nobody has missed them. On ACROSS-TRACK r they are not: at
every position, "predict each track's average level" already reproduces the fact that some
tracks are globally higher than others, and across-track r rewards exactly that. **So the
across-track floor is NOT zero, has never been measured, and every across-track number in the
report and deck is currently quoted without one.** If a trained arm does not clear trackmean_loo
by a wide margin, its across-track r is measuring track-level offsets, not tissue specificity.

MATCHED BY CONSTRUCTION, NOT BY HAND. --like points at a real run directory; windows, manifest,
fasta, split, geometry, loci seed, track subset and the position subsample are all taken from
the same code path eval_many_loci.py uses, so these floors sit on the the common axis axis by construction
rather than by my remembering to pass matching flags.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, required=True,
                    help="a run dir whose last.pt supplies windows/manifest/fasta/geometry, so "
                         "the baseline is matched to the benchmark by construction")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-loci", type=int, default=96)
    ap.add_argument("--pos-per-locus", type=int, default=6000)
    ap.add_argument("--loci-seed", type=int, default=0)
    ap.add_argument("--track-subset", type=int, default=48)
    ap.add_argument("--pool-bp", type=int, default=1,
                    help="average targets down the POSITION axis by this factor before the "
                         "baselines are built and before any statistic. 1 = native (default), "
                         "so every existing floor reproduces byte-for-byte. 128 puts the floor "
                         "on E0's resolution -- AG's accessibility heads are 128 bp-native, "
                         "and a 128 bp arm compared to a 1 bp floor is the same error the "
                         "--pool-bp flag in eval_many_loci.py exists to prevent.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from agtracks.dataset import GenomeWindowDataset

    t = torch.load(a.like / "last.pt", map_location="cpu", weights_only=False)["args"]
    bs = int(t.get("bin_size", 1))
    ds = GenomeWindowDataset(Path(t["windows"]), Path(t["manifest"]), Path(t["fasta"]),
                              split=a.split, bin_size=bs,
                              max_tracks=t.get("max_tracks"), track_seed=t.get("track_seed", 0))
    T = ds.n_tracks()
    TS = int(a.track_subset)
    if TS > T:
        raise SystemExit(f"--track-subset {TS} > {T}")
    Teff = TS or T
    _raw = [x.get("track_mean") for x in ds.tracks]
    means = np.clip(np.array([m if (m is not None and m > 0) else 1.0 for m in _raw],
                             dtype=np.float32), 1e-3, None)

    # IDENTICAL loci selection to eval_many_loci.py -- same rng, same call order.
    rng = np.random.default_rng(a.loci_seed)
    n_avail = len(ds)
    idx = (np.arange(n_avail) if n_avail <= a.n_loci
           else np.sort(rng.choice(n_avail, a.n_loci, replace=False)))
    n_loci = len(idx)
    if n_loci < 2:
        raise SystemExit("leave-one-out needs >= 2 loci")

    def load(w: int) -> np.ndarray:
        """(L, Teff) targets in the SAME units the trainer and evaluator use (y * 1/track_mean)."""
        _, y = ds[int(w)]
        Y = (np.asarray(y, dtype=np.float64) / means[None, :])[:, :Teff]
        # 🚨 POOL HERE, AT THE SINGLE ENTRY POINT FOR Y, so sumY, L, sum_scalar, the live
        # filter and the strata are ALL at the pooled resolution. Pooling later would build
        # the baselines at one resolution and score them at another.
        # ✅ This is exact, not an approximation: both baselines are LINEAR in Y (leave-one-
        # out means), and mean-pooling commutes with them -- pool(meanprofile_loo) equals
        # meanprofile_loo computed on pooled Y, and trackmean_loo is a per-track scalar that
        # mean-pooling leaves unchanged.
        if a.pool_bp > 1:
            L0 = (Y.shape[0] // a.pool_bp) * a.pool_bp
            Y = Y[:L0].reshape(L0 // a.pool_bp, a.pool_bp, Y.shape[1]).mean(1)
        return Y

    # ---- pass 1: accumulate the sum over loci. 1,048,576 x 48 float64 = 402 MB. ----
    print(f"pass 1/2: summing {n_loci} loci x {Teff} tracks", flush=True)
    sumY = None
    for c, w in enumerate(idx):
        Y = load(w)
        sumY = Y.copy() if sumY is None else sumY + Y
        if (c + 1) % 16 == 0:
            print(f"  {c+1}/{n_loci}", flush=True)
    L = sumY.shape[0]
    sum_scalar = sumY.sum(0)                       # (Teff,) total per track over all loci

    # ---- pass 2: leave-one-out prediction per locus, metrics identical to eval_many_loci ----
    print("pass 2/2: leave-one-out scoring", flush=True)
    stats = {k: {"n": 0, "sx": np.zeros(Teff), "sy": np.zeros(Teff), "sxx": np.zeros(Teff),
                 "syy": np.zeros(Teff), "sxy": np.zeros(Teff), "acr": [],
                 "sumP": 0.0, "tP": np.zeros(Teff)}
             for k in ("meanprofile_loo", "trackmean_loo")}
    sumY_total = 0.0
    tY = np.zeros(Teff)
    n_live_tot = n_pos_tot = 0
    # ONE generator, and the subsample indices are shared by both baselines and reproduce the
    # sequence eval_many_loci.py draws -- the live set depends only on Y, not on the prediction.
    gen = torch.Generator().manual_seed(a.loci_seed)
    # ASSAY STRATA -- see eval_many_loci.py. The POOLED floor is not usable: measured on the
    # constant-per-track baseline it reads +0.534 on val and -0.244 on test, because track_mean
    # normalisation is genome-wide while RNA sits at 2.49x its genome-wide mean on chr9/10 vs
    # 1.22x on chr8. Within an assay group the floor is stable, so the strata are the floors
    # that can actually be quoted.
    _grp = ["RNA" if "RNA" in str(x.get("assay", "")).upper() else "ACC"
            for x in ds.tracks[:Teff]]
    strata = {g: [i for i, gg in enumerate(_grp) if gg == g] for g in sorted(set(_grp))}
    for st_ in stats.values():
        st_["acr_by_assay"] = {g: [] for g in strata}

    for c, w in enumerate(idx):
        Y = load(w)
        preds = {
            "meanprofile_loo": (sumY - Y) / (n_loci - 1),
            # constant per track: mean over the other loci AND all positions, broadcast
            "trackmean_loo": np.broadcast_to(
                ((sum_scalar - Y.sum(0)) / ((n_loci - 1) * L))[None, :], Y.shape),
        }
        sumY_total += float(Y.sum()); tY += Y.sum(0)
        Yt = torch.from_numpy(Y)
        live = Yt.std(1) > 1e-6
        n_live_tot += int(live.sum()); n_pos_tot += int(live.numel())
        sel = None
        if int(live.sum()):
            nl = int(live.sum())
            sel = (torch.randperm(nl, generator=gen)[:a.pos_per_locus]
                   if nl > a.pos_per_locus else torch.arange(nl))
        for k, Pn in preds.items():
            s = stats[k]
            P = torch.from_numpy(np.ascontiguousarray(Pn))
            s["n"] += P.shape[0]
            s["sx"] += P.sum(0).numpy(); s["sy"] += Yt.sum(0).numpy()
            s["sxx"] += (P * P).sum(0).numpy(); s["syy"] += (Yt * Yt).sum(0).numpy()
            s["sxy"] += (P * Yt).sum(0).numpy()
            s["sumP"] += float(P.sum()); s["tP"] += P.sum(0).numpy()
            if sel is not None:
                Pl, Yl = P[live][sel], Yt[live][sel]
                da = Pl - Pl.mean(1, keepdim=True); db = Yl - Yl.mean(1, keepdim=True)
                s["acr"].append(((da * db).sum(1) /
                                 (da.norm(dim=1) * db.norm(dim=1)).clamp_min(1e-8)))
            for g, cols in strata.items():
                if len(cols) < 2:
                    continue
                Pg, Yg = P[:, cols], Yt[:, cols]
                lg = Yg.std(1) > 1e-6
                Pg, Yg = Pg[lg], Yg[lg]
                if not Pg.shape[0]:
                    continue
                if Pg.shape[0] > a.pos_per_locus:
                    Pg, Yg = Pg[:a.pos_per_locus], Yg[:a.pos_per_locus]
                dc = Pg - Pg.mean(1, keepdim=True); dd = Yg - Yg.mean(1, keepdim=True)
                s["acr_by_assay"][g].append(((dc * dd).sum(1) /
                                             (dc.norm(dim=1) * dd.norm(dim=1)).clamp_min(1e-8)))
        if (c + 1) % 16 == 0:
            print(f"  {c+1}/{n_loci}", flush=True)

    res = {"baseline_for": a.like.name, "split": a.split, "n_loci": n_loci,
           "n_loci_available": int(n_avail), "n_tracks": Teff, "track_subset": TS,
           "manifest": Path(t["manifest"]).name, "output_bp": int(L),
           "pool_bp": int(a.pool_bp),
           "n_positions_live": n_live_tot, "n_positions_total": n_pos_tot,
           "loci_seed": a.loci_seed, "baselines": {}}
    for k, s in stats.items():
        n = s["n"]
        num = n * s["sxy"] - s["sx"] * s["sy"]
        den = np.sqrt(np.maximum(n * s["sxx"] - s["sx"] ** 2, 0) *
                      np.maximum(n * s["syy"] - s["sy"] ** 2, 0))
        r_model = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        A = torch.cat(s["acr"]) if s["acr"] else torch.zeros(1)
        res["baselines"][k] = {
            "r_across_median": float(A.median()),
            "r_across_iqr": [float(A.quantile(.25)), float(A.quantile(.75))],
            "r_model_median": float(np.median(r_model)),
            "n_across_samples": int(A.numel()),
            "cov_ratio": s["sumP"] / max(sumY_total, 1e-6),
            "r_across_by_assay": {
                g: (lambda A_: {"median": float(A_.median()), "n_tracks": len(strata[g]),
                                "n_samples": int(A_.numel())})(torch.cat(v))
                for g, v in s["acr_by_assay"].items() if v},
            "r_model_per_track": [float(x) for x in r_model],
            "cov_ratio_per_track": [float(p / q) if q > 0 else None
                                    for p, q in zip(s["tP"], tY)],
        }
    res["track_labels"] = [{"i": i, "assay": x.get("assay"), "tissue": x.get("tissue"),
                            "study": x.get("study"), "track_id": x.get("track_id")}
                           for i, x in enumerate(ds.tracks[:Teff])]
    out = a.out or (None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["baselines"].items():
        byas = " ".join(f"{g}={d['median']:+.4f}"
                        for g, d in sorted(v.get("r_across_by_assay", {}).items()))
        print(f"{k:18s} pooled={v['r_across_median']:+.4f} {byas} "
              f"within={v['r_model_median']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
