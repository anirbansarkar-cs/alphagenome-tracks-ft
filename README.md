# agtracks — fine-tune AlphaGenome to predict multi-track coverage profiles

One shared AlphaGenome trunk, one multi-channel head, **T tracks predicted per base pair**.

This is the profile counterpart to the encoder-only, sequence-to-scalar fine-tuning in
[`alphagenome_FT_MPRA`](https://github.com/Al-Murphy/alphagenome_FT_MPRA) and
[`alphagenome-encoder-ft`](https://github.com/MasayukiNagai/alphagenome_encoder_ft). Those pool the
length axis away and regress one or two scalars per sequence. This one keeps the length axis and
fits a coverage profile over many tracks at once.

| | encoder-only repos | this repo |
|---|---|---|
| output | 1–2 scalars per sequence | profile over T tracks, per base |
| length axis | pooled away | preserved, up to 1,048,576 bp |
| decoder | unused | used; `--bin-size 128` skips it |
| objective | MSE | AlphaGenome's segmented Poisson + multinomial |

Nothing here is species-specific. The species lives in three input files and one flag.

## Install

```bash
pip install -r requirements.txt
pip install "alphagenome-pytorch @ git+https://github.com/genomicsxai/alphagenome-pytorch"
pip install -e .

python tests/test_losses.py        # CPU only, no data, no checkpoint — run this first
```

`requirements.txt` lists the runtime dependencies and records the exact version set this code is
run on, in case a loose lower bound bites. Two things it deliberately cannot install for you:

- **`alphagenome-pytorch`** is not on PyPI, hence the separate line above. `agtracks.train_seqpar`
  additionally needs `alphagenome_pytorch.sequence_parallel`; if that import fails your version
  predates sequence-parallel support. The single-GPU and DDP paths do not need it.
- **The AlphaGenome weights.** Supply your own and pass `--checkpoint`. Nothing is shipped here,
  and the trainer refuses to start from random init on purpose: `RMSBatchNorm` never updates
  `running_var`, so an uninitialised model trains with no normalisation and reports no error.

Verified working set: Python 3.11.15, torch 2.11.0+cu130, numpy 2.4.3, safetensors 0.7.0,
pyfaidx 0.9.0.3, pyBigWig 0.3.25, einops 0.8.0.

## Run

```bash
python -m agtracks.train \
    --checkpoint alphagenome.safetensors \
    --windows windows.json --manifest tracks.json --fasta genome.fa \
    --organism-index 0 \
    --trainable all --lr 3e-4 --steps 12000 \
    --out runs/first
```

`--organism-index`: AlphaGenome ships **two** organism rows, `0 = human`, `1 = mouse`. Pick the
one your data is. A third species needs a wider embedding table than the published checkpoint has,
which is outside this repo's scope.

`--steps` is absolute and `--resume` restores optimizer state and the step counter, so budgets
chain rather than restart:

```bash
python -m agtracks.train ... --steps 6000  --out runs/first/a
python -m agtracks.train ... --steps 12000 --resume runs/first/a/last.pt --out runs/first/b
```

## Getting many tracks onto one GPU

There is no trick, only three things that have to be right.

1. **One trunk, one multi-channel head.** Each extra track costs its head channel plus
   activations, not another model. This is what the repo is for.
2. **Gradient checkpointing**, on by default (`--no-grad-ckpt` to disable).
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, which is load-bearing rather than a
   nicety. The trainer sets it and logs it. On one 93 GiB card, 1,100 output channels at 1 bp fit
   with it and OOM without.

Measured ceilings on a single 93 GiB H100, for sizing:

| output resolution | channels that fit | note |
|---|---|---|
| 1 bp | ~1,100 | activations scale with length × tracks |
| 128 bp | ~12,000 in 43.8 GiB | `--bin-size 128` skips the decoder entirely |

If you have tens of tracks you will not come near either limit.

### Past one GPU: DDP raises throughput, sequence parallelism raises capacity

This is the one place where the obvious choice is the wrong one, so it is worth stating plainly.

| | what it shards | throughput | track ceiling at 1 bp |
|---|---|---|---|
| **DDP** | the batch | scales with GPUs | **~1,100, and more GPUs never raise it** |
| **sequence parallelism** | the sequence | 1.23× from 2 GPUs to 4 | 2 GPUs ~4,000 · 4 GPUs **8,000+** |

**DDP replicates the model.** Each rank pays weights plus the *full* activation for the *full*
sequence, so adding GPUs buys throughput and **can never raise the track ceiling**. It stays at
roughly 1,100 channels at 1 bp however many GPUs you add.

**Sequence parallelism shards the sequence**, so per-GPU activation and head memory fall with world
size and the ceiling scales with it:

| world size | channels at 1 bp | per-track cost |
|---|---|---|
| 1 | ~1,100 | — |
| 2 | ~4,000 | 14.0 MB |
| 4 | **8,000+** | 7.1 MB |

8,000 is a measured floor: it fits, 12,000 OOMs, and we did not bisect between them. Per-track cost
halves exactly from world 2 to 4. For reference, AlphaGenome's own full 4,992-track configuration
runs at native 1 Mb on 4 GPUs in 53.94 GiB, with 42% headroom.

⚠️ **It buys capacity, not speed.** Throughput from world 2 to 4 measured **1.23×**, not 2×. So:

- tracks fit on one GPU → plain single-GPU
- want it faster and tracks already fit → **DDP** (`torchrun -m agtracks.train`)
- track count is the binding constraint → **sequence parallelism**
  (`torchrun -m agtracks.train_seqpar`)

`scripts/probe_track_ceiling.py` measures the ceiling for your own card and world size before you
commit to a track count:

```bash
torchrun --nproc_per_node=4 scripts/probe_track_ceiling.py --tracks 512 1024 2048 4096
```

It reports which counts fit and which OOM, so bracket your intended number rather than guessing.

### Running it

```bash
torchrun --nproc_per_node=4 -m agtracks.train_seqpar \
    --windows W.json --manifest M.json --fasta genome.fa \
    --checkpoint AG.safetensors --organism-index 0 \
    --steps 2000 --out runs/sp
```

Three things to know before the first run:

1. **`output_bp` must divide by the world size.** Each rank takes an equal slice of the output
   positions, so a window index with `output_bp = 131072` works on 1, 2 or 4 GPUs but not 3. The
   dataset raises a clear error rather than silently truncating.
2. **Leave `--replicate-to` alone.** It fans the output channels out by *repeating* tracks, to
   measure a capacity ceiling without needing that many real bigWigs. It defaults to off. Setting
   it trains on duplicated targets and the model is not meaningful.
3. **`--overlap-high` must be a multiple of 128**, because the encoder downsamples the 1 bp shard
   by 128 before the gather trims in the 128 bp domain. The low-resolution overlap is derived from
   it automatically, so there is one number to set, not two. It raises if you get it wrong.

To convince yourself the sharded path is doing the same thing as the unsharded one, `--no-seqpar`
runs identical data and schedule on one GPU so the loss trajectories can be compared side by side.

Correctness of the sharded path against the unsharded one: loss exact to 1.3e-6, gradient cosine
0.9996. ⚠️ Do not validate a long sharded run by weight identity — agreement degrades
superlinearly because Adam normalises by gradient magnitude. Compare validation metrics.

Two design points, in case you are tempted to wire sequence parallelism up yourself. Both are easy
to get wrong and neither fails loudly.

1. **The sampler is not a `DistributedSampler`.** Under DDP each rank takes a *different window*.
   Under sequence parallelism every rank takes the *same window* and a different slice of its
   *positions*. Ranks walk an identical, identically-seeded window order, and the dataset serves
   each rank only its positional shard through `shard_rank` / `shard_world`. Hand it a
   `DistributedSampler` and each rank shards a different window, which trains on nonsense while
   looking healthy.
2. **The target is expanded in chunks.** A wide 1 bp target runs to gigabytes per rank in fp32, and
   materialising it whole doubles the head's own output cost for nothing. The loss accumulates over
   track chunks, which is exactly equivalent because a mean over all channels equals the
   size-weighted mean of per-chunk means.

Predictions are also clamped before the implicit `exp()`: with a randomly-initialised head the raw
values overflow fp32 and give `inf` loss with `NaN` gradients.

**Your bottleneck will probably be I/O, not the GPU.** Before staging bigWigs to local tmpfs, a
1 Mb training step here was 86% data loading — 11.0 s of a 12.8 s step. Fixing that alone was an
8.5× end-to-end speedup, and it is independent of the model.

## The loss

`agtracks.losses.ag_loss` is AlphaGenome's published objective: Poisson on 8 segment sums plus a
5× multinomial over positions within each segment. So the Poisson term sets **level** and the
multinomial sets **shape** — and the multinomial is exactly scale-invariant, meaning the Poisson
term is the only thing anchoring absolute magnitude. `tests/test_losses.py` verifies that property
directly, since it is the one most easily broken by a well-meaning edit.

It is shipped because it is the published objective, which makes it the sensible starting point
and the sensible thing to compare against. **Whether it suits your readout is an empirical
question for your own data.** To swap it, replace the single call marked `🔁 SWAP YOUR OWN LOSS IN
HERE` in `train.py`; the contract is `(rate, target) -> (scalar, {name: float})` with `rate >= 0`
and shape `(B, L, T)`.

`--target-clip ag` additionally applies AlphaGenome's target scaling step 2, the sqrt smooth-clip.
Default is `none`.

## Evaluating

Scripts under `scripts/`, all optional but the first is strongly recommended:

- `eval_many_loci.py` — score on many held-out loci with per-assay strata. **Use this rather than
  your run's own validation number.** A per-run validation figure computed on a handful of fixed
  windows is a trap; averaging more validations does not widen it, and it will happily rank your
  models wrongly.
- `baselines.py` — sequence-free floors. A model that does not beat a constant-per-track predictor
  should be caught in minutes, not weeks.
- `eval_profiles.py` — shape metrics plus sequence nulls: dinucleotide shuffle, GC-only, and
  reverse-complement consistency. The nulls are how you show the model reads sequence at all.
- `eval_variant_effect.py` + `variant_report.py` — in-silico mutagenesis with the level confound
  controlled. Read that script's docstring before trusting any variant number.

## Layout

```
src/agtracks/   dataset.py  losses.py  train.py  train_seqpar.py
scripts/        eval_many_loci.py  baselines.py  eval_profiles.py
                eval_variant_effect.py  variant_report.py  build_window_index.py
                probe_track_ceiling.py
tests/          test_losses.py         # CPU, no data
requirements.txt, requirements-dev.txt
docs/           DATA_SCHEMA.md  GOTCHAS.md
```

See `docs/DATA_SCHEMA.md` for the three input files and `docs/GOTCHAS.md` before debugging
anything surprising.

## Licence

MIT, see `LICENSE`. Chosen to match
[`alphagenome_FT_MPRA`](https://github.com/Al-Murphy/alphagenome_FT_MPRA), the fine-tuning work
this descends from. The imported `alphagenome-pytorch` is Apache 2.0, which is permissive and
places no constraint on this choice. No AlphaGenome code is copied: `src/agtracks/losses.py` is an
independent implementation of the objective described in the paper's Methods, with their pseudocode
quoted in a docstring for attribution.

⚠️ **This licence does not cover model weights.** No weights are distributed here. If you later
want to share a *fine-tuned checkpoint*, that is governed by the terms attached to the AlphaGenome
weights you started from, not by this licence. Check those before distributing a trained artefact.
