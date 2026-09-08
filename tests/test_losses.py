#!/usr/bin/env python3
"""CPU property tests for agtracks.losses -- no GPU, no data, no checkpoint needed.

Run this first after installing. It validates the rate parameterisation, AlphaGenome's target
scaling, and the segmented objective (including its scale-invariance, which is the property most
easily broken by a well-meaning edit) without touching a GPU or any of your files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from agtracks import losses as L

torch.manual_seed(0)
ok = True
def check(name, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    ok = ok and bool(cond)

print("1. rate parameterisation -- the 1.0 floor")
head_out = torch.rand(2, 64, 3) * 3.0            # softplus output: non-negative, some near 0
r_exp    = L.rate_from_head(head_out, "exp")
r_direct = L.rate_from_head(head_out, "direct")
check("old 'exp' path floors every rate at 1.0", float(r_exp.min()) >= 1.0,
      "min=%.4f" % r_exp.min())
check("'direct' can represent background", float(r_direct.min()) < 0.05,
      "min=%.6f" % r_direct.min())
check("'direct' is the identity on a non-negative head output",
      torch.allclose(r_direct, head_out))

print("\n2. AlphaGenome loss vs an independent reference implementation")
B, S, T, NS = 2, 256, 4, 8
rate = torch.rand(B, S, T) * 5 + 0.01
tgt  = torch.poisson(torch.rand(B, S, T) * 5)
tot, pois, mult = L.ag_loss(rate, tgt, n_segments=NS)
# reference: literal transcription of the paper pseudocode, then the same normalisation
seg = S // NS
x  = rate.reshape(B, NS, seg, T); y = tgt.reshape(B, NS, seg, T)
sp = x.sum(2); sy = y.sum(2)
ref_p = (sp - sy * torch.log(sp + 1e-7)).sum()
ref_m = (-y * torch.log(x / (sp.unsqueeze(2) + 1e-7) + 1e-7)).sum()
den = B * S * T
check("poisson part matches paper pseudocode",
      torch.allclose(pois, ref_p / seg / den, atol=1e-5))
check("multinomial part matches paper pseudocode",
      torch.allclose(mult, 5.0 * ref_m / den, atol=1e-4))
check("total is the sum of the two parts", torch.allclose(tot, pois + mult, atol=1e-6))
print("     observed magnitudes:  poisson %.3e   multinomial %.3e   ratio 1:%.1e"
      % (pois, mult, float(mult / pois.abs())))

print("\n3. AlphaGenome loss is minimised at the truth")
truth = torch.poisson(torch.rand(1, S, T) * 8) + 0.01
at_truth, _, _ = L.ag_loss(truth, truth)
worse = [float(L.ag_loss(truth * f, truth)[0]) for f in (0.5, 0.8, 1.25, 2.0)]
check("perturbing the prediction raises the loss", all(w > float(at_truth) for w in worse),
      "truth=%.4f  perturbed=%s" % (at_truth, [round(w, 4) for w in worse]))
shuf = truth[:, torch.randperm(S)]
check("shuffling the PROFILE raises the loss (shape term is live)",
      float(L.ag_loss(shuf, truth)[0]) > float(at_truth))

print("\n4. numerical safety")
z = torch.zeros(1, 64, 2)
t = torch.poisson(torch.rand(1, 64, 2) * 3)
tot, _, _ = L.ag_loss(z, t)
check("all-zero prediction gives a finite loss", torch.isfinite(tot), "loss=%.3f" % tot)
big = torch.full((1, 64, 2), 1e6)
check("huge prediction gives a finite loss", torch.isfinite(L.ag_loss(big, t)[0]))
check("zero target gives a finite loss", torch.isfinite(L.ag_loss(rate[:1], torch.zeros(1, S, T))[0]))

print("\n5. scale-invariance: WHY the small Poisson term is not a bug")
yy = torch.poisson(torch.rand(1, 8192, 8) * 2)
pp = yy.clamp_min(1e-3)
mults = [float(L.ag_loss(pp * c, yy)[2]) for c in (0.25, 0.5, 1.0, 2.0, 4.0)]
poiss = [float(L.ag_loss(pp * c, yy)[1]) for c in (0.25, 0.5, 1.0, 2.0, 4.0)]
check("multinomial is EXACTLY invariant to uniform scaling", len(set(mults)) == 1,
      "%.6e at every scale" % mults[0])
check("poisson DOES respond to scale (it is the only term that can)",
      len(set(poiss)) == len(poiss))
check("poisson is minimised nearest the correct scale", poiss.index(min(poiss)) == 2,
      "argmin at factor %s" % [0.25, 0.5, 1.0, 2.0, 4.0][poiss.index(min(poiss))])
w1, w100 = L.ag_loss(pp, yy), L.ag_loss(pp, yy, poisson_weight=100.0)
check("poisson_weight leaves the shape term untouched", float(w100[2]) == float(w1[2]))
check("poisson_weight scales the anchor linearly",
      abs(float(w100[1]) - 100 * float(w1[1])) < 1e-4 * abs(100 * float(w1[1])),
      "1x=%.6e  100x=%.6e" % (w1[1], w100[1]))

print("\n6. term balance at REAL scale (1 Mb x 48 tracks, 8 segments)")
S3, T3 = 1 << 20, 48
tr = torch.poisson(torch.rand(1, S3, T3, dtype=torch.float32) * 0.5)
_, p3, m3 = L.ag_loss(tr.clamp_min(1e-3), tr)
print("     at the truth:      poisson %+.4e   multinomial %+.4e" % (p3, m3))
_, p4, m4 = L.ag_loss(torch.full_like(tr, 0.25), tr)
print("     flat prediction:   poisson %+.4e   multinomial %+.4e" % (p4, m4))
print("     -> multinomial moves %.3e, poisson moves %.3e  (ratio %.0f:1)"
      % (abs(m4 - m3), abs(p4 - p3), abs(m4 - m3) / max(abs(p4 - p3), 1e-12)))
check("at real scale the shape term dominates the gradient signal",
      abs(m4 - m3) > abs(p4 - p3))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
