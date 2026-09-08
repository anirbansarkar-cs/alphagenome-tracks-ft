# Gotchas

Things that fail quietly. Each one cost real time somewhere.

## The head emits a rate, not a log-rate

`GenomeTracksHead(..., return_scaled=True)` returns `softplus(x) * softplus(scale)`, already
non-negative. AlphaGenome consumes it directly. Exponentiating it gives an effective rate of
`exp(softplus(·)) >= 1`, so the model **cannot represent background at all**: predictions come out
with min and median both 1.0000 against a median target of 0.0, and nothing errors.
`losses.rate_from_head` is the single place that decision lives. Leave `--rate-param direct`.

## Do not judge a run by its own validation number

A per-run validation figure computed on a handful of fixed windows is not a held-out estimate, and
averaging more validations does not widen it — it averages training noise over the same loci. Use
`scripts/eval_many_loci.py` on many loci. Loci, not positions, are the unit of evidence: positions
inside one window are autocorrelated over roughly a kilobase, so a large position count from few
windows is not independent support.

## Two correlations, and they can disagree

- **within-window r** runs down one track and scores profile *shape*. Blind to level.
- **across-track r** runs across tracks at one position and depends on their *relative levels*.

They can rank your settings differently, and both are real; they measure different things. Decide
which one your downstream use needs *before* tuning, or you will tune against the wrong one.

## `expandable_segments` is load-bearing

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the difference between fitting and OOM at the
top of the track range, not a micro-optimisation. The trainer sets it and logs it, so the log
records what was actually in force.

## Never validate a long run by weight identity

Two mathematically equivalent code paths diverge in weight space superlinearly, because Adam
normalises by gradient magnitude: gradient cosine can be 0.9996 while weight cosine is
0.999999998, and the gap grows with steps. Compare **validation metrics**, not weights.

## `--steps` is absolute

`--resume` restores optimizer state and the step counter. Resuming a 6,000-step checkpoint with
`--steps 12000` trains 6,000 more; passing `--steps 6000` again trains zero. This is deliberate, so
budgets chain into a curve rather than restarting.

## Coverage ratio is your calibration readout

Total predicted over total observed. The multinomial term cannot see overall scale, so this is the
only number that says whether the Poisson anchor is doing its job. 1.0 is calibrated. A value far
below 1.0 means the level term is being outvoted.

## Score every checkpoint you already have before training anything new

Scoring a checkpoint costs minutes; training one costs hours. Chained runs leave intermediate
checkpoints on disk that are easy to never look at, and an answer already paid for is the cheapest
answer available.

## If you add an organism row

Initialise it from an existing row, not randomly. The trunk expects organism vectors of a
particular scale and direction; a random vector is a large out-of-distribution perturbation at the
first layer, and fine-tuning has to undo it before it can learn anything.

## DDP will not raise your track ceiling

DDP replicates the model, so every rank pays weights plus the full activation for the full
sequence. Adding GPUs buys throughput; the number of output channels that fit is unchanged. If you
are OOMing on track count, more GPUs under DDP is not the fix — sequence parallelism is
(`agtracks.train_seqpar`), and it scales the ceiling roughly linearly with world size while buying
almost no speed (1.23× from 2 GPUs to 4).
