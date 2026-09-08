# Licence

**Not yet chosen — pick one before making this repository public.** There is no legal obstacle to a
permissive licence; this file exists so the choice is made deliberately rather than by omission.

## What was checked

- **`alphagenome-pytorch`**, the only dependency that could have constrained the choice, is
  **Apache 2.0**. Permissive, no copyleft, so it places no requirement on this repository's
  licence. Apache 2.0 is therefore the natural default here, since it matches the dependency and
  carries an explicit patent grant.
- **No AlphaGenome code is copied.** `src/agtracks/losses.py` is an independent PyTorch
  implementation of the objective described in the AlphaGenome paper's Methods. The paper's
  pseudocode appears only as a short quotation in a docstring, for attribution.
- **No model weights and no genomic data are included**, and `.gitignore` excludes
  `*.safetensors`, bigWigs and FASTA so they cannot be committed by accident.

## The one thing that is genuinely separate

Nothing above governs **a fine-tuned checkpoint**. If you later want to share weights produced with
this code, that is decided by the terms attached to the AlphaGenome weights you started from, not
by this repository's licence. Check those terms before distributing any trained artefact.

Institutional policy may also apply to the licence choice; that is worth confirming before pushing.
