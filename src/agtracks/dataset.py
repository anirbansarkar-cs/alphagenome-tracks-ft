#!/usr/bin/env python3
"""Training dataset: window index + track manifest -> (one-hot sequence, per-track targets).

Turns three artifacts into batches: a window index (JSON), a track manifest (JSON), and a FASTA.
Nothing here is species-specific -- the species lives entirely in those three files plus the
`--organism-index` you pass the trainer.

Three checks exist because the corresponding mistake is easy to make and fails silently:

1. SEQUENCE IS KEYED BY (species, assembly), NOT BY SPECIES. The same species can appear on two
   assemblies whose chromosomes differ by megabases. A loader keyed by species alone would serve
   the wrong sequence for some tracks and never raise.

2. CONTIG NAMES ARE PER TRACK, NOT PER SPECIES. One source may use `1..22`, another `chr1..chr22`,
   another `Chr01..Chr22`. The manifest carries `contig_style` per track for exactly this reason,
   and a track whose chromosome cannot be resolved is a HARD ERROR rather than a silently-skipped
   block of zeros. Contig naming is the single most common source of quiet corruption here.

3. THE SPLIT LIVES IN THE WINDOW INDEX, not in this loader. Whole-chromosome splits make the
   leakage rule structural; re-deriving it here would be a chance to get it wrong.

Targets are RAW COVERAGE, not log-transformed: the AlphaGenome objective expects counts, and
transforming here would silently change what the model fits. bigWig gaps become 0.0, which for
coverage means "no reads", not "unknown".

Usage:
    python -m agtracks.dataset --windows W.json --manifest M.json --fasta genome.fa --self-test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CONTIG_PREFIXES = ("chr", "chrom")

_BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def norm_contig(name: str) -> str:
    """`Chr01`/`chr1`/`1` -> `1`. Anchored regex, not lstrip (which strips characters).

    Extend CONTIG_PREFIXES if your assembly uses a different prefix. Anything unmatched is
    returned lowercased and zero-stripped, so `X`/`Y`/`MT` pass through unchanged.
    """
    n = re.sub(r"^(" + "|".join(CONTIG_PREFIXES) + r")", "", name.strip().lower())
    return n.lstrip("0") or n


class GenomeWindowDataset:
    """Map-style dataset over the prebuilt window index.

    Deliberately NOT a torch.utils.data.Dataset subclass at import time -- torch is imported
    lazily so this module can be inspected and self-tested without a GPU environment.
    """

    def __init__(self, windows_json: Path, manifest_json: Path, fasta: Path,
                 split: str = "train", include_controls: bool = True,
                 assays: tuple[str, ...] | None = None, require_local: bool = True,
                 max_tracks: int | None = None, track_seed: int = 0,
                 replicate_to: int | None = None, bin_size: int = 1,
                 shard_rank: int = 0, shard_world: int = 1):
        # Output resolution. AlphaGenome's own heads predict ChIP at 128 bp ONLY (chip_tf and
        # chip_histone have convs.128 and no convs.1 in the released checkpoint); 1 bp is
        # reserved for DNase/ATAC/RNA/CAGE/PROCAP. Predicting everything at 1 bp costs 128x the
        # memory per track and is what put our OOM wall at ~1,100 channels.
        if bin_size < 1:
            raise ValueError(f"bin_size must be >= 1, got {bin_size}")
        self.bin_size = bin_size
        wi = json.loads(Path(windows_json).read_text())
        self.input_bp = wi["input_bp"]
        self.output_bp = wi["output_bp"]
        self.crop = wi["crop"]
        if self.output_bp % bin_size:
            raise ValueError(f"output_bp {self.output_bp} not divisible by bin_size {bin_size}")
        self.n_bins = self.output_bp // bin_size
        # POSITIONAL SHARDING, for sequence parallelism. Under DDP each rank takes a different
        # WINDOW; under sequence parallelism every rank takes the SAME window and a different
        # slice of its positions. Reading only this rank's slice keeps the target tensor
        # world_size times smaller and avoids reading bigwig ranges we would discard.
        if shard_world < 1 or not (0 <= shard_rank < shard_world):
            raise ValueError(f"bad shard {shard_rank}/{shard_world}")
        if self.output_bp % shard_world:
            raise ValueError(f"output_bp {self.output_bp} not divisible by world {shard_world}")
        self.shard_rank, self.shard_world = shard_rank, shard_world
        self.shard_bp = self.output_bp // shard_world
        if self.shard_bp % bin_size:
            raise ValueError(f"shard {self.shard_bp} not divisible by bin_size {bin_size}")
        self.shard_bins = self.shard_bp // bin_size
        self.return_index = False
        self.windows = [w for w in wi["windows"] if w["split"] == split]
        if not self.windows:
            raise ValueError(f"no windows for split={split!r}")

        man = json.loads(Path(manifest_json).read_text())
        tracks = [t for t in man if include_controls or not t.get("is_control")]
        if assays:
            tracks = [t for t in tracks if t["assay"] in assays]
        # A track with no local file cannot be read. Fail loudly at construction rather than
        # producing zeros at batch time -- a silently absent track looks like a flat signal,
        # which the model will happily learn.
        resolved, missing = [], []
        for t in tracks:
            p = t.get("path") or t.get("local")
            if p and Path(p).exists():
                resolved.append({**t, "_path": p})
            else:
                missing.append(t["track_id"])
        if missing and require_local:
            raise FileNotFoundError(
                f"{len(missing)}/{len(tracks)} tracks have no local bigwig yet, e.g. "
                f"{missing[:4]}. Pass require_local=False to train on the available subset, "
                f"but record WHICH tracks were used.")
        # Track subsetting, for the output-channel scaling study. STRATIFIED BY ASSAY, and that
        # is not a nicety: an assay-skewed panel (say 800 ChIP against 100 ATAC) makes a
        # uniform sample of 50 tracks would be ~96% ChIP-Seq and could contain no DNase at all.
        # A scaling curve measured on a set whose composition drifts with n is measuring two
        # things at once. Round-robin across assays keeps the mix as close to fixed as the
        # counts allow, and the seeded shuffle makes the choice reproducible and recorded.
        if max_tracks is not None and max_tracks < len(resolved):
            import random
            by_assay: dict[str, list] = {}
            for t in resolved:
                by_assay.setdefault(t.get("assay", "?"), []).append(t)
            rng = random.Random(track_seed)
            for v in by_assay.values():
                v.sort(key=lambda t: t["track_id"])      # deterministic before shuffling
                rng.shuffle(v)
            picked, order = [], sorted(by_assay)          # rarest assay first, so it survives
            order.sort(key=lambda a: len(by_assay[a]))
            while len(picked) < max_tracks:
                progressed = False
                for a in order:
                    if by_assay[a] and len(picked) < max_tracks:
                        picked.append(by_assay[a].pop()); progressed = True
                if not progressed:
                    break
            picked.sort(key=lambda t: t["track_id"])      # stable channel order across ranks
            resolved = picked

        # CAPACITY-PROBE ONLY: pad the channel count past the number of real tracks by
        # REPEATING tracks. With ~1,000 distinct bigWigs the OOM ceiling above that
        # cannot be measured with real data. GPU memory and GPU compute depend only on the
        # head's output shape (B, L, T) -- a duplicated channel costs exactly what a distinct
        # one costs -- so this measures the ceiling faithfully. What it does NOT do is produce
        # a trainable model: duplicated channels mean a duplicated loss term. Never use this
        # for a real run; it exists to find where the card breaks.
        self.replicated_from = None
        self._src_idx = list(range(len(resolved)))
        if replicate_to is not None and replicate_to > len(resolved):
            n = len(resolved)
            if n == 0:
                raise ValueError("cannot replicate from zero resolved tracks")
            self._src_idx = [i % n for i in range(replicate_to)]
            self.replicated_from = n
            resolved = [resolved[i] for i in self._src_idx]

        self.tracks = resolved
        self.missing = missing
        self.fasta_path = str(fasta)
        self._fa = None
        self._bw: dict[str, object] = {}
        self._name_cache: dict[str, dict[str, str]] = {}
        self._fa_names: dict[str, str] | None = None

    # ---- lazy handles: opened per worker, never shared across processes -------------
    def _fasta(self):
        if self._fa is None:
            from pyfaidx import Fasta
            self._fa = Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        return self._fa

    def _fasta_contig(self, canon: str) -> str:
        """Resolve canonical `1` to whatever the FASTA calls chromosome 1.

        🚨 The window index stores NORMALISED names (`1`, `15`) so --val-chroms/--test-chroms
        match across assemblies, but a FASTA may use `chr15` (hg38) or `Chr01`. Looking the raw
        canonical name up in pyfaidx raised KeyError: '15' on hg38. Bigwig names were already
        resolved this way by _track_contig; the FASTA was not, and a FASTA may happen to
        use bare names -- so the gap stayed invisible until a second assembly arrived.
        """
        if self._fa_names is None:
            self._fa_names = {norm_contig(k): k for k in self._fasta().keys()}
        if canon not in self._fa_names:
            raise KeyError(f"FASTA {self.fasta_path} has no contig matching canonical {canon!r}; "
                           f"has {sorted(self._fa_names)[:6]}")
        return self._fa_names[canon]

    def _bigwig(self, path: str):
        if path not in self._bw:
            import pyBigWig
            self._bw[path] = pyBigWig.open(path)
        return self._bw[path]

    def _track_contig(self, path: str, canon: str) -> str:
        """Resolve canonical `1` to whatever THIS bigwig calls chromosome 1."""
        if path not in self._name_cache:
            bw = self._bigwig(path)
            self._name_cache[path] = {norm_contig(k): k for k in bw.chroms()}
        m = self._name_cache[path]
        if canon not in m:
            raise KeyError(
                f"track {path} has no contig matching canonical {canon!r}; "
                f"has {sorted(m)[:6]}. Contig naming differs PER TRACK -- check contig_style.")
        return m[canon]

    def __len__(self) -> int:
        return len(self.windows)

    def n_tracks(self) -> int:
        return len(self.tracks)

    def __getitem__(self, i: int):
        import numpy as np
        w = self.windows[i]
        chrom = w["chrom"]

        seq = str(self._fasta()[self._fasta_contig(chrom)][w["in_start"]:w["in_end"]])
        if len(seq) != self.input_bp:
            raise ValueError(f"window {i}: got {len(seq)} bp, expected {self.input_bp}")
        # one-hot (L,4); N and any other ambiguity code stay all-zero, which is the honest
        # encoding for "unknown base" -- not a uniform 0.25, which asserts equal evidence.
        oh = np.zeros((self.input_bp, 4), dtype=np.float32)
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        for b, j in _BASES.items():
            oh[arr == ord(b), j] = 1.0

        n_uniq = self.replicated_from if self.replicated_from else len(self.tracks)
        s0 = w["out_start"] + self.shard_rank * self.shard_bp
        s1 = s0 + self.shard_bp
        base = np.zeros((self.shard_bins, n_uniq), dtype=np.float32)
        for k in range(n_uniq):
            t = self.tracks[k]
            bw = self._bigwig(t["_path"])
            name = self._track_contig(t["_path"], chrom)
            v = bw.values(name, s0, s1, numpy=True)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if self.bin_size > 1:
                # MEAN over each bin, matching how coverage is aggregated -- a sum would change
                # the target's scale by bin_size and silently rescale the Poisson rate.
                v = v.reshape(self.shard_bins, self.bin_size).mean(axis=1)
            base[:, k] = v
        if self.replicated_from is None:
            out = (oh, base)
        else:
            # fancy-index, not tile: _src_idx is the authoritative channel->source map
            out = (oh, np.ascontiguousarray(base[:, self._src_idx]))
        # The gene-level tissue loss needs to know WHICH window this is, to look up gene
        # spans. Off by default so every existing caller keeps its 2-tuple.
        return out + (i,) if getattr(self, "return_index", False) else out

    def track_table(self):
        return [{"i": i, "track_id": t["track_id"], "assay": t["assay"],
                 "is_control": bool(t.get("is_control")), "tissue": t.get("tissue")}
                for i, t in enumerate(self.tracks)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    ds = GenomeWindowDataset(args.windows, args.manifest, args.fasta,
                              split=args.split, require_local=False)
    print(f"split={args.split}  windows={len(ds):,}  tracks_available={ds.n_tracks()}"
          f"  tracks_missing={len(ds.missing)}")
    print(f"  input {ds.input_bp:,}  output {ds.output_bp:,}  crop {ds.crop:,}")
    if ds.missing:
        print(f"  ⚠️ not yet produced: {len(ds.missing)} e.g. {ds.missing[:5]}")
    if not args.self_test:
        return 0
    if ds.n_tracks() == 0:
        print("no local tracks yet — nothing to sample"); return 0

    import numpy as np
    for i in (0, len(ds) // 2, len(ds) - 1)[:args.n]:
        x, y = ds[i]
        w = ds.windows[i]
        acgt = float(x.sum() / x.shape[0])
        print(f"  win {i:>7} {w['chrom']}:{w['out_start']:,}  x={x.shape} "
              f"ACGT_frac={acgt:.3f}  y={y.shape} mean={y.mean():.3f} max={y.max():.1f} "
              f"nonzero_tracks={(y.sum(0) > 0).sum()}/{y.shape[1]}")
    print("\ntrack order (index -> id) for the head:")
    for r in ds.track_table()[:6]:
        print(f"  {r['i']:>3} {r['track_id']:<30}{r['assay']:<24}"
              f"{'CONTROL' if r['is_control'] else ''}")
    print("SELFTEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
