# Input files

Three files, plus one flag. Nothing else determines the species.

## 1. Window index — `--windows`

JSON. Carries the geometry and the train/val/test split, so the loader never re-derives either.

```json
{
  "input_bp":  1048576,
  "output_bp": 131072,
  "crop":      458752,
  "windows": [
    {"chrom": "1", "in_start": 0, "in_end": 1048576,
     "out_start": 458752, "split": "train"}
  ]
}
```

- `crop` is the offset from `in_start` to `out_start`. Prediction at output index *j* corresponds
  to input position `crop + j`.
- `split` is one of `train`, `val`, `test`.
- **Put the split here, not in your loader.** Whole-chromosome splits make the leakage rule
  structural; anything else needs the input-window overlap rule applied when the index is built.

`scripts/build_window_index.py` builds one from a FASTA, with `--val-chroms` / `--test-chroms` for
chromosome-level splits, or `--sections N` for AlphaGenome-style within-chromosome splits that drop
any val/test window whose 1 Mb *input* window overlaps a training window's input.

## 2. Track manifest — `--manifest`

JSON list, one entry per output channel. Channel order is manifest order.

```json
[
  {"track_id": "dnase_rep1",
   "assay":    "DNase",
   "path":     "/abs/path/dnase_rep1.bw",
   "tissue":   "leaf",
   "track_mean": 1.842,
   "contig_style": "chr",
   "is_control": false}
]
```

| field | required | purpose |
|---|---|---|
| `track_id` | yes | identity, used in reports |
| `assay` | yes | groups tracks into strata for per-assay scoring |
| `path` (or `local`) | yes | absolute path to the bigWig |
| `track_mean` | recommended | genome-wide mean, used to normalise targets. Missing or ≤0 falls back to 1.0, which silently changes what the model fits |
| `contig_style` | if needed | how this track names chromosomes, e.g. `chr` vs bare |
| `tissue`, `is_control` | optional | carried through to reports |

**Use absolute paths.** A relative path resolves against whatever the working directory happens to
be, which fails quietly by loading the wrong file rather than none.

**`contig_style` is per track, not per manifest.** Sources disagree: `1`, `chr1`, `Chr01`. A track
whose chromosome cannot be resolved is a hard error, never a silently-skipped block of zeros. This
is the most common source of quiet corruption.

## 3. Reference FASTA — `--fasta`

Indexed for random access (`pyfaidx`). Must be the assembly the bigWigs were built against. Same
species on a different assembly is a real failure mode: chromosomes can differ by megabases, and a
loader keyed on species alone would serve the wrong sequence and never raise.

## 4. `--organism-index`

`0` for human, `1` for mouse. These are the two rows the published AlphaGenome checkpoint ships.

## Targets

Read as **raw coverage**, not log-transformed: the objective expects counts. bigWig gaps become
`0.0`, which for coverage means "no reads", not "unknown". Do not pre-transform your bigWigs.
