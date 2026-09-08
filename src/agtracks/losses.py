#!/usr/bin/env python3
"""AlphaGenome's own multi-track output objective, in PyTorch.

This module implements the loss described in the AlphaGenome paper's Methods, section
"RNA-seq, CAGE, ATAC, DNase and PRO-Cap Output Heads". Their formulation:

    def multinomial_loss(x, targets, multinomial_resolution):
      x       = x.reshape((-1, multinomial_resolution, x.shape[-1]))
      targets = targets.reshape((-1, multinomial_resolution, targets.shape[-1]))
      sum_pred, sum_target = Sum(x, axis=1), Sum(targets, axis=1)
      poisson_loss    = Sum(sum_pred - sum_target * Log(sum_pred + 1e-7))
      multinomial_prob = x / (sum_pred + 1e-7)
      positional_loss = Sum(-targets * Log(multinomial_prob + 1e-7))
      return (poisson_loss / multinomial_resolution + 5.0 * positional_loss)

So the Poisson term constrains only the SEGMENT TOTAL -- 8 segments over the sequence -- and the
profile shape is carried by a multinomial weighted 5.0. That split matters: the multinomial is
exactly scale-invariant (d/d log s = 0 for a per-segment rescale s), so it sets shape and the
Poisson term is the only thing that sets absolute level.

⚠️ THIS IS ONE LOSS, NOT THE RIGHT LOSS FOR YOUR TASK. It is shipped because it is the published
   one, so it is the sensible starting point and the sensible thing to compare against. Whether it
   suits your readout is an empirical question you should answer on your own data -- see
   `ag_loss`'s docstring for the signature to implement if you want to swap it out.

🚨 THE RATE PARAMETERISATION, which is easy to get wrong. `GenomeTracksHead(..., return_scaled=True)`
   returns `softplus(x) * softplus(scale)`, i.e. a NON-NEGATIVE RATE. AlphaGenome consumes that
   directly, since its loss takes `Log(sum_pred)`. Treating it as a LOG-rate and exponentiating it
   gives an effective rate of exp(softplus(.)) >= 1, so the model cannot represent background at
   all: on a shipped checkpoint that produced predictions with min and median both 1.0000 against a
   median target of 0.0. `rate_from_head` is the single place that decision lives. Leave it at
   'direct' unless you know your head emits logs.
"""
from __future__ import annotations

import torch

EPS = 1e-7


def rate_from_head(pred, mode: str = "direct"):
    """Turn the head's output into a Poisson RATE.

    'direct' -- AlphaGenome's contract: the head already emits softplus(x)*softplus(scale) >= 0,
                which IS the rate. This is correct and is the default.
    'exp'    -- the historical (wrong) behaviour, kept only so pre-fix checkpoints stay
                evaluable. Floors every prediction at exp(0) = 1.
    """
    if mode == "direct":
        return pred.clamp_min(0.0)
    if mode == "exp":
        return torch.exp(torch.clamp(pred, max=20.0))
    raise ValueError(f"unknown rate parameterisation {mode!r}")


# --------------------------------------------------------------------- AG target scaling
# AlphaGenome p.32. We had the FIRST step (divide by track mean) and were missing the second.
#
#   def targets_scaling(targets, track_means, apply_squashing):
#     targets = targets / track_means
#     if apply_squashing:            # Applied RNA-seq tracks only.
#       targets = targets ** 0.75
#     return Where(targets > 10.0, 2 * Sqrt(x * 10.0) - 10.0, targets)
#
# ⚠️ READ THE INDENTATION: `apply_squashing` gates ONLY the `** 0.75` power. The sqrt smooth-clip
# is OUTSIDE that branch and therefore applies to EVERY track. The paper's ChIP-seq section
# confirms it — "target scaling (without squashing)" means without the power but WITH the clip.
# (`Sqrt(x * 10.0)` in their snippet is a typo for `Sqrt(targets * 10.0)`; x is not in scope.)
#
# WHY IT MATTERS. The multinomial is a y-weighted sum of -log p, so it is dominated by the
# heaviest bins; an unclipped tail lets a handful of positions set the objective. This is the
# leading candidate for the measured cov ~= 0.15 (see EXPERIMENT_PLAN.md s16).
#
# The transform is continuous and monotone at the knee: t=10 maps to 2*sqrt(100)-10 = 10, so
# `t > 10` and `f(t) > 10` describe the same set and the inverse below is exact.
_CLIP_KNEE = 10.0


def targets_scaling(target, rna_mask=None, clip: bool = True, squash_rna: bool = True):
    """AG's target transform, applied AFTER division by track_mean.

    target   : (..., T) already divided by its per-track mean.
    rna_mask : bool tensor (T,) marking RNA-seq columns, or None to skip the power term.
    """
    t = target
    if squash_rna and rna_mask is not None and bool(rna_mask.any()):
        m = rna_mask.to(t.device)
        t = torch.where(m, t.clamp_min(0.0) ** 0.75, t)
    if clip:
        k = _CLIP_KNEE
        t = torch.where(t > k, 2.0 * torch.sqrt(t.clamp_min(0.0) * k) - k, t)
    return t


def predictions_scaling(x, rna_mask=None, clip: bool = True, squash_rna: bool = True):
    """Exact inverse of `targets_scaling`, for reporting in original units."""
    if clip:
        k = _CLIP_KNEE
        x = torch.where(x > k, (x + k) ** 2 / (4.0 * k), x)
    if squash_rna and rna_mask is not None and bool(rna_mask.any()):
        m = rna_mask.to(x.device)
        x = torch.where(m, x.clamp_min(0.0) ** (1.0 / 0.75), x)
    return x

def ag_loss(rate, target, n_segments: int = 8, multinomial_weight: float = 5.0,
                   poisson_weight: float = 1.0, denom=None):
    """AlphaGenome/Borzoi loss. rate/target are (B, S, T), rate non-negative.

    Returns (total, poisson_part, multinomial_part) with every part normalised per element.
    AlphaGenome returns a SUM; summing over ~50M elements would make the learning rate
    meaningless here, so all three are divided by B*S*T. That is a uniform rescale -- it does
    not change the RELATIVE weight of the two terms, which is the thing that matters.

    Splitting into 8 segments rather than scoring the whole sequence at once is AlphaGenome's
    own choice: they report smaller segments empirically degraded performance, and a single
    whole-sequence multinomial is numerically unstable at 2**20 bins.
    """
    B, S, T = rate.shape
    seg = S // n_segments
    if seg == 0:
        raise ValueError(f"sequence length {S} shorter than n_segments {n_segments}")
    use = seg * n_segments
    p = rate[:, :use].reshape(B, n_segments, seg, T)
    y = target[:, :use].reshape(B, n_segments, seg, T)

    sum_p = p.sum(2)                                       # (B, n_segments, T)
    sum_y = y.sum(2)
    poisson = (sum_p - sum_y * torch.log(sum_p + EPS)).sum()

    prob = p / (sum_p.unsqueeze(2) + EPS)
    positional = (-y * torch.log(prob + EPS)).sum()

    # NOTE ON THE POISSON TERM'S SIZE. It is ~5 orders of magnitude below the multinomial and
    # that is CORRECT, not a bug: the multinomial is EXACTLY invariant to scaling the prediction
    # uniformly (verified -- bit-identical across a 16x scale sweep), so it contributes ZERO
    # gradient along the scale direction. The Poisson is the only term that can set total
    # coverage at all, so its magnitude relative to the multinomial is not the thing to judge it
    # by. `poisson_weight` exists to strengthen it if the total turns out to converge too slowly
    # at AlphaGenome's own weighting -- a deliberate deviation, defaulting to faithful (1.0).
    denom = (B * use * T) if denom is None else denom
    pois_n = poisson_weight * poisson / seg / denom
    mult_n = multinomial_weight * positional / denom
    return pois_n + mult_n, pois_n.detach(), mult_n.detach()
