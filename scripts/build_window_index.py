#!/usr/bin/env python3
"""Build the training window index: tiling, train/val/test splits, and N-gap exclusion.

Three things this settles, each of which is a silent-failure risk if left to the loader:

1. TILING. Windows are tiled by OUTPUT span, so consecutive windows are adjacent in output
   space and overlap in input space by 2*crop -- the Borzoi/AlphaGenome scheme. No position is
   predicted twice; the overlap is context reuse only.

2. SPLITS. Whole chromosomes are assigned to train/val/test. With chromosome-level splits,
   AlphaGenome's leakage rule (drop any val/test interval whose 1 Mb INPUT window overlaps a
   training interval's input window) is satisfied STRUCTURALLY -- windows on different
   chromosomes cannot overlap at any context length. That is worth stating explicitly, because
   the rule becomes load-bearing the moment anyone switches to within-chromosome sections, and
   it would then be easy to forget. `--sections` implements that mode WITH the rule applied.

3. N-GAP EXCLUSION. Assembly gaps are runs of N. A window sitting in a gap trains the model to
   predict signal from no sequence, and inflates the apparent dataset size. Windows whose input
   span exceeds --max-n-frac ambiguous bases are dropped, and the count is reported rather than
   quietly absorbed.

Usage:
    python build_window_index.py --input-bp 16384 --output-bp 8192
    python build_window_index.py --sections 8          # AG-style within-chromosome folds
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path



def load_chromosomes(registry: Path, assembly: str) -> tuple[dict[str, int], str]:
    reg = json.loads(registry.read_text())
    for v in reg.values():
        if v["assembly"] == assembly:
            return v["chromosomes"], v["fasta"]
    raise SystemExit(f"assembly {assembly!r} not in {registry}")


def _norm(name: str) -> str:
    """`chr1`/`Chr01`/`1` -> `1`. Same rule as agtracks.dataset.norm_contig."""
    import re
    n = re.sub(r"^(chr|bd)", "", name.strip().lower())
    return n.lstrip("0") or n


def n_profile(fasta: Path, chroms: dict[str, int], bin_bp: int) -> dict[str, list[int]]:
    """Count ambiguous bases per bin, per chromosome. One pass over the FASTA."""
    want = set(chroms)
    prof = {c: [0] * (chroms[c] // bin_bp + 1) for c in chroms}
    cur, pos = None, 0
    # Accept EITHER a gzipped or a plain FASTA. Many assemblies ship as .fa.gz; some are kept
    # uncompressed because pyfaidx needs plain or BGZF, and a second 950 MB gzipped copy purely
    # to satisfy this reader would be waste. Detected by magic bytes, not by file extension --
    # `.fa` naming is not a guarantee of anything.
    with open(fasta, "rb") as _probe:
        _gz = _probe.read(2) == b"\x1f\x8b"
    with (gzip.open(fasta, "rt") if _gz else open(fasta, "rt")) as fh:
        for line in fh:
            if line.startswith(">"):
                # 🚨 NORMALISE. The registry stores chromosomes as bare names ("1", "8") so
                # that --val-chroms/--test-chroms match, but FASTA headers may be ">chr1"
                # (hg38) or ">Chr01". Matching the raw header against `want` silently skipped
                # EVERY line of hg38: dropped_high_n came back 0 on a genome whose chr1 is
                # 7.42% N, which would have put pure-centromere windows into training.
                cur = _norm(line[1:].split()[0])
                cur = cur if cur in want else None
                pos = 0
                continue
            if cur is None:
                continue
            s = line.strip().upper()
            if "N" in s:
                # attribute Ns to bins; lines are short so per-line granularity is fine
                for i, ch in enumerate(s):
                    if ch == "N":
                        b = (pos + i) // bin_bp
                        if b < len(prof[cur]):
                            prof[cur][b] += 1
            pos += len(s)
    return prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--assembly", required=True,
                    help="assembly name, recorded in the index so a loader can key on it")
    ap.add_argument("--input-bp", type=int, default=16384)
    ap.add_argument("--output-bp", type=int, default=8192)
    ap.add_argument("--max-n-frac", type=float, default=0.10)
    ap.add_argument("--stride", type=int, default=0,
                    help="bp between consecutive window STARTS. 0 = tile by output span "
                         "(non-overlapping). AlphaGenome's own recipe spaces 1 Mb windows "
                         "~196,608 bp apart (81%% overlap), taken from Borzoi's target "
                         "intervals -- so for 1 Mb use --stride 196608.")
    ap.add_argument("--val-chroms", nargs="*", default=["8"])
    ap.add_argument("--test-chroms", nargs="*", default=["9", "10"])
    ap.add_argument("--sections", type=int, default=0,
                    help="if >0, split WITHIN chromosomes into N sections and apply AG's "
                         "input-window leakage rule instead of splitting by chromosome")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.input_bp < args.output_bp or (args.input_bp - args.output_bp) % 2:
        raise SystemExit("input-bp must be >= output-bp and differ by an even amount")
    crop = (args.input_bp - args.output_bp) // 2
    stride = args.stride or args.output_bp

    chroms, fasta = load_chromosomes(args.registry, args.assembly)
    print(f"assembly {args.assembly}: {len(chroms)} chromosomes, "
          f"{sum(chroms.values()):,} bp")
    print(f"input {args.input_bp:,}  output {args.output_bp:,}  crop {crop:,} each side")
    print(f"stride {stride:,} between window starts"
          f"{' (OVERLAPPING, AlphaGenome-style)' if stride < args.output_bp else ''}\n")

    print("scanning FASTA for assembly gaps (N runs)...", flush=True)
    prof = n_profile(Path(fasta), chroms, args.output_bp)

    windows, dropped_n, dropped_edge = [], 0, 0
    for c, length in chroms.items():
        nbins = len(prof[c])
        for k in range((length - args.output_bp) // stride + 1):
            out_start = k * stride
            in_start, in_end = out_start - crop, out_start + args.output_bp + crop
            if in_start < 0 or in_end > length:
                dropped_edge += 1          # input window would run off the chromosome
                continue
            b = out_start // args.output_bp
            ncount = prof[c][b] if b < nbins else 0
            if ncount / args.output_bp > args.max_n_frac:
                dropped_n += 1
                continue
            windows.append({"chrom": c, "out_start": out_start,
                            "out_end": out_start + args.output_bp,
                            "in_start": in_start, "in_end": in_end})

    if args.sections:
        # AG-style: split each chromosome into sections, then DROP any val/test window whose
        # INPUT span overlaps a train window's INPUT span. Without this, up to `crop` bp of
        # training context leaks across every section boundary.
        for w in windows:
            sec = (w["out_start"] * args.sections) // chroms[w["chrom"]]
            w["split"] = "train" if sec < args.sections - 2 else (
                "val" if sec == args.sections - 2 else "test")
        train = {}
        for w in windows:
            if w["split"] == "train":
                train.setdefault(w["chrom"], []).append((w["in_start"], w["in_end"]))
        kept, leaked = [], 0
        for w in windows:
            if w["split"] == "train":
                kept.append(w); continue
            if any(s < w["in_end"] and w["in_start"] < e
                   for s, e in train.get(w["chrom"], [])):
                leaked += 1; continue
            kept.append(w)
        print(f"  leakage rule dropped {leaked} val/test windows overlapping train context")
        windows = kept
    else:
        val, test = set(args.val_chroms), set(args.test_chroms)
        for w in windows:
            w["split"] = "val" if w["chrom"] in val else ("test" if w["chrom"] in test else "train")
        print("  split by whole chromosome ⇒ AG's input-window leakage rule is satisfied "
              "STRUCTURALLY (windows on different chromosomes cannot overlap)")

    from collections import Counter
    cnt = Counter(w["split"] for w in windows)
    print(f"\ndropped: {dropped_n} windows >{args.max_n_frac:.0%} N, "
          f"{dropped_edge} at chromosome edges")
    print(f"{'split':<8}{'windows':>10}{'bp covered':>16}{'% of set':>10}")
    tot = sum(cnt.values())
    for s in ("train", "val", "test"):
        n = cnt.get(s, 0)
        print(f"{s:<8}{n:>10,}{n*args.output_bp:>16,}{100*n/max(tot,1):>9.1f}%")
    print(f"{'TOTAL':<8}{tot:>10,}{tot*args.output_bp:>16,}")

    args.out.write_text(json.dumps(
        {"assembly": args.assembly, "input_bp": args.input_bp, "output_bp": args.output_bp,
         "crop": crop, "max_n_frac": args.max_n_frac,
         "val_chroms": args.val_chroms, "test_chroms": args.test_chroms,
         "sections": args.sections, "stride": stride,
         "dropped_high_n": dropped_n, "dropped_edge": dropped_edge,
         "counts": dict(cnt), "windows": windows}, indent=None) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
