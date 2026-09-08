#!/usr/bin/env python3
"""Fine-tune AlphaGenome to predict multi-track coverage profiles at base resolution.

One shared trunk, one multi-channel `GenomeTracksHead`, T tracks out per position. The species is
not in the code: it is in the window index, the track manifest, the FASTA, and `--organism-index`.

Usage:
    python -m agtracks.train --checkpoint AG.safetensors \
        --windows W.json --manifest M.json --fasta genome.fa \
        --organism-index 0 --trainable all --lr 3e-4 --steps 18000 --out runs/my_run
    torchrun --nproc_per_node=4 -m agtracks.train ...     # DDP
    python -m agtracks.train --smoke ...                  # a few steps, sanity only

`--organism-index`: AlphaGenome ships two organism rows, **0 = human, 1 = mouse**. Pick the one
your data is. A third species needs a wider embedding table than the published checkpoint has.

`--steps` is ABSOLUTE and `--resume` restores optimizer state and the step counter, so budgets
chain: run to 6000, then resume the same run with --steps 12000 to continue rather than restart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agtracks import losses as PL


def build_model(num_tracks: int, num_organisms: int, checkpoint: Path | None,
                grad_ckpt: bool, device: str, track_means=None):
    import torch
    import torch.nn as nn
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.heads import GenomeTracksHead
    from safetensors.torch import load_file

    # AlphaGenome's constructor hardcodes splice-head track counts for num_organisms in
    # {1, 2} and leaves `splice_usage_tracks_per_organism` UNDEFINED otherwise, so
    # AlphaGenome(num_organisms=3) raises UnboundLocalError before we ever get to delete the
    # splice heads. Build with 2 -- which works -- then widen the organism embeddings in the
    # live modules. This avoids patching a shared conda env that other people use.
    model = AlphaGenome(num_organisms=2, gradient_checkpointing=grad_ckpt)
    if num_organisms > 2:
        n_widened = 0
        for mod in model.modules():
            emb = getattr(mod, "organism_embed", None)
            if isinstance(emb, nn.Embedding) and emb.num_embeddings < num_organisms:
                new = nn.Embedding(num_organisms, emb.embedding_dim)
                with torch.no_grad():
                    new.weight[: emb.num_embeddings] = emb.weight
                    # new organism starts as a copy of HUMAN (row 0), not noise: the trunk
                    # expects organism vectors of a particular scale, and a random vector is a
                    # large out-of-distribution perturbation at the first layer.
                    new.weight[emb.num_embeddings:] = emb.weight[0]
                mod.organism_embed = new
                n_widened += 1
        model.num_organisms = num_organisms
        print(f"  widened {n_widened} organism embeddings 2 -> {num_organisms}")
    if checkpoint is None:
        raise SystemExit(
            "refusing to train from random init: RMSBatchNorm never updates running_var, so an "
            "uninitialised model trains with no normalisation and reports no error. "
            "Pass --checkpoint.")
    sd = load_file(str(checkpoint))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    got = [k for k in sd if "organism_embed" in k]
    for k in got:
        if sd[k].shape[0] < num_organisms:
            raise SystemExit(f"{k} has {sd[k].shape[0]} organism rows, need {num_organisms}; "
                             f"run expand_organism_embedding.py")
    print(f"  loaded {len(sd)} tensors; {len(missing)} missing, {len(unexpected)} unexpected")

    model.heads = nn.ModuleDict()
    model.contact_maps_head = None
    model.splice_sites_classification_head = None
    model.splice_sites_usage_head = None
    model.splice_sites_junction_head = None
    for p in model.embedder_pair.parameters():
        p.requires_grad_(False)

    head = GenomeTracksHead(in_channels=None, num_tracks=num_tracks,
                            resolutions=(1, 128), num_organisms=num_organisms,
                            track_means=track_means)
    return model.to(device), head.to(device)


# AG's own track heads, by assay. Only these three are relevant to us: chip_histone/chip_tf are
# 128 bp-only, and cage/procap may have no counterpart in your manifest.
AG_HEAD_FOR_ASSAY = {"DNASE": "dnase", "ATAC": "atac", "RNA": "rna_seq"}


def init_head_from_ag(head, tracks, checkpoint, seed=0, verbose=True):
    """E1a/E1b: initialise our track head from AlphaGenome's PRETRAINED head rows.

    E2 -- the control -- initialises the SAME head randomly. Holding the trainable set fixed and
    changing ONLY the initialisation is what isolates INIT from CAPACITY, so this must be run at
    E2's learning rate or the comparison measures two things at once.

    🚨 ROWS ARE CHOSEN WITHOUT LOOKING AT OUR DATA. The tempting alternative -- reuse E0's
    best-matching AG track for each of your tracks -- selects rows on a MAX STATISTIC computed on
    an evaluation split, which leaks that split into the initialisation and would make E1b beat
    E2 for a reason that has nothing to do with pretrained weights. Rows are drawn by a seeded
    RNG, WITHOUT REPLACEMENT within each assay group, and never touch a target.

    AG's head is (num_organisms=2, num_tracks, in_channels); ours is (3, T, in_channels). We copy
    AG's HUMAN row (organism 0) into ALL of our organism rows -- the same convention build_model
    already uses when it widens organism_embed.

    ⚠️ `track_means` is deliberately NOT copied: it is a buffer already holding OUR per-track
    means, and overwriting it with AG's human means would silently rescale every target.
    """
    import random
    import torch
    from safetensors.torch import load_file

    sd = load_file(str(checkpoint))
    ours = head.state_dict()

    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tracks):
        a = str(t.get("assay", "")).upper()
        name = next((v for k, v in AG_HEAD_FOR_ASSAY.items() if a.startswith(k)), None)
        if name is None:
            raise SystemExit(f"--head-init ag: track {i} assay {t.get('assay')!r} maps to no AG head")
        groups.setdefault(name, []).append(i)

    rng = random.Random(seed)
    picked: dict[int, tuple[str, int]] = {}
    for name in sorted(groups):
        idxs = groups[name]
        n_rows = sd[f"heads.{name}.convs.128.weight"].shape[1]
        if len(idxs) > n_rows:
            raise SystemExit(f"--head-init ag: need {len(idxs)} rows from AG head {name}, has {n_rows}")
        for i, r in zip(idxs, rng.sample(range(n_rows), len(idxs))):
            picked[i] = (name, r)

    for res in ("1", "128"):
        for kind in ("weight", "bias"):
            key = f"convs.{res}.{kind}"
            if key not in ours:
                raise SystemExit(f"--head-init ag: our head has no {key}")
            dst = ours[key].clone()
            for i, (name, r) in picked.items():
                src = sd[f"heads.{name}.convs.{res}.{kind}"]
                if kind == "weight" and src.shape[-1] != dst.shape[-1]:
                    raise SystemExit(f"{key}: AG in_channels {src.shape[-1]} != ours {dst.shape[-1]}")
                dst[:, i] = src[0, r].to(dst.dtype)
            ours[key] = dst
        key = f"residual_scales.{res}"
        if key not in ours:
            raise SystemExit(f"--head-init ag: our head has no {key}")
        dst = ours[key].clone()
        for i, (name, r) in picked.items():
            dst[:, i] = sd[f"heads.{name}.residual_scales.{res}"][0, r].to(dst.dtype)
        ours[key] = dst

    missing, unexpected = head.load_state_dict(ours, strict=True)
    if verbose:
        per = {n: len(v) for n, v in sorted(groups.items())}
        print(f"  --head-init ag: seeded {len(picked)} track(s) from AG head rows {per} "
              f"(seed={seed}, without replacement, data never consulted)", flush=True)
    return picked


def poisson_nll(pred_log, target):
    """Poisson NLL with the model emitting log-rate. Constant term dropped."""
    import torch
    rate = torch.exp(torch.clamp(pred_log, max=20.0))
    return (rate - target * pred_log).mean()


def multinomial_shape(pred_log, target, eps=1e-8):
    """Cross-entropy between predicted and observed PROFILE SHAPE, per track.

    Separating shape from counts matters: a model can nail total coverage and place it wrongly.
    pred_log/target: (B, L, T).
    """
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(pred_log, dim=1)
    tot = target.sum(dim=1, keepdim=True).clamp_min(eps)
    q = target / tot
    return -(q * logp).sum(dim=1).mean()


def main() -> int:
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from agtracks.dataset import GenomeWindowDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    required=True)
    # AlphaGenome standard: 1 Mb, stride 196,608 (its own recipe). The 16 kb index remains at
    # A smaller output window trains faster, but 1 Mb is what AlphaGenome was
    # pretrained at and the receptive field assumes it.
    # windows, and supervision is per-POSITION regardless.
    ap.add_argument("--windows", type=Path,
                    required=True)
    ap.add_argument("--manifest", type=Path,
                    required=True)
    ap.add_argument("--fasta", type=Path,
                    required=True)
    ap.add_argument("--organism-index", type=int, default=2,
                    help="which organism row to condition on. AlphaGenome ships 0=human, 1=mouse")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="train on a stratified subset of output channels. This is the knob for "
                         "the scaling study: how many tracks fit before memory or throughput "
                         "breaks. Subset is stratified by assay so the mix does not drift with n.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch/numpy/random. Without this every run drew a different "
                         "head init and data order, so a 2-seed comparison was neither "
                         "reproducible nor a controlled estimate of seed spread.")
    ap.add_argument("--track-seed", type=int, default=0,
                    help="seed for the track subset; recorded in the run config")
    ap.add_argument("--num-organisms", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per GPU. MEASURED: 1 Mb x 128 tracks = 73.5 GB at B=1 with "
                         "--grad-ckpt; B=2 OOMs on a 93 GB H100.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a checkpoint written by this trainer: restores model, "
                         "head, Adam moments (matched BY PARAMETER NAME) and the step counter. "
                         "Params that were not trainable in the saved run start with fresh "
                         "moments, which is what E4's staged schedule needs.")
    ap.add_argument("--trainable", choices=["head", "head+decoder", "all"], default="all",
                    help="which parameter groups get gradients -- the E-series axis. "
                         "'head' = E2 probe (the control to beat). "
                         "'head+decoder' = E3 decoder-only. "
                         "'all' = E5 full fine-tuning, and the historical default, so existing "
                         "recipes are unchanged. E4 runs these in sequence from one checkpoint.")
    ap.add_argument("--val-select", choices=["prefix", "spread", "rotate"], default="rotate",
                    help="WHICH val windows to score. 🚨 `prefix` -- the historical behaviour and "
                         "still the default -- scores the FIRST --val-batches windows of the "
                         "split, THE SAME ONES EVERY VALIDATION (the loader is shuffle=False and "
                         "validate() breaks early). At batch-size 1 that is 8 loci, 0.11%% of "
                         "W1a's split and 2.7%% of the 1 Mb split, and averaging validations does "
                         "NOT widen it. It is why W1a's 1 Mb 'win' was real-looking: arms with "
                         "different window indices start at different genomic positions, and the "
                         "4k and 1 Mb arms shared ZERO of their 8 windows. "
                         "`spread` takes --val-batches windows evenly spaced across the whole "
                         "split (deterministic, identical every validation and across arms that "
                         "share an index) -- same cost, representative loci. "
                         "`rotate` advances the spread block each validation, so a 60-validation "
                         "run covers many distinct loci and an average over the last 20 rests on "
                         "20x the loci -- best coverage, noisier per validation. "
                         "⚠️ DEFAULT LEFT AT `prefix` ON PURPOSE: E4 spawns a fresh process per "
                         "phase, so flipping the default mid-run would give one staged run P1 on "
                         "one sampling rule and P2/P3 on another. Cut over deliberately, and "
                         "note that histories across the change are NOT comparable.")
    ap.add_argument("--head-init", choices=["random", "ag"], default="random",
                    help="random = E2's control. ag = E1a/E1b: seed the track head from "
                         "AlphaGenome's pretrained head rows, assay-matched, rows drawn by a "
                         "seeded RNG WITHOUT consulting any target. Must be paired with E2's "
                         "LR or it stops isolating init from capacity.")
    ap.add_argument("--head-init-seed", type=int, default=-1,
                    help="RNG for AG row selection; -1 = use --seed.")
    ap.add_argument("--organism-embed-trainable", type=int, default=1, choices=[0, 1],
                    help="keep organism_embed trainable even when the backbone is frozen. "
                         "DEFAULT 1 AND THAT MATTERS: build_model widens the table 2->3 and "
                         "initialises row 2 as a COPY OF THE HUMAN ROW, so freezing it leaves "
                         "your species conditioned as whatever row you picked, with the "
                         "species conditioning inert. ~12k params; counted as head.")
    ap.add_argument("--target-clip", choices=["none", "ag"], default="none",
                    help="AG's targets_scaling step 2. 'ag' applies the sqrt smooth-clip "
                         "Where(t>10, 2*sqrt(10t)-10, t) to EVERY track and the **0.75 power to "
                         "RNA-seq columns only, after the existing division by track_mean. "
                         "'none' is what we have been doing -- division only, which is NOT what "
                         "AlphaGenome does (paper p.32). Default stays 'none' so this flag "
                         "changes nothing until a run opts in.")
    ap.add_argument("--rate-param", choices=["direct", "exp"], default="direct",
                    help="how to turn the head output into a Poisson rate. 'direct' is "
                         "AlphaGenome's contract (the head already emits softplus>=0). 'exp' "
                         "is the old, wrong behaviour that floored every prediction at 1.0; "
                         "keep it ONLY to re-evaluate pre-fix checkpoints.")
    ap.add_argument("--n-segments", type=int, default=8)
    ap.add_argument("--multinomial-weight", type=float, default=5.0)
    ap.add_argument("--poisson-weight", type=float, default=1.0,
                    help="AlphaGenome-faithful is 1.0. The Poisson term is tiny next to the "
                         "multinomial by design -- the multinomial is exactly scale-invariant, so "
                         "the Poisson is the ONLY term that sets total coverage. Raise this only "
                         "if the measured coverage ratio fails to converge.")
    ap.add_argument("--bin-size", type=int, default=1, choices=[1, 128],
                    help="output resolution in bp. 1 = per-base (memory-hungry); 128 = the "
                         "resolution AlphaGenome's own ChIP heads use. The head supports "
                         "exactly these two.")
    ap.add_argument("--replicate-to", type=int, default=None,
                    help="CAPACITY PROBE ONLY: pad channels past the real track count by "
                         "repeating tracks, to find the OOM ceiling. Not a trainable config.")
    ap.add_argument("--grad-ckpt", action="store_true", default=True,
                    help="required at 1 Mb: B=1 without it OOMs")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25,
                    help="steps between progress lines. Peak GPU memory is reported with\neach one -- that number IS the deliverable of the scaling ladder, and without it a run\nthat merely SURVIVES tells us nothing about how much headroom was left.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.val_every, args.val_batches = 12, 6, 4

    # SEED EVERYTHING that varies between otherwise-identical runs: the new head's init, the
    # training window order, and any dropout. Under DDP each rank gets seed+rank so ranks do not
    # draw identical data order, while the run as a whole stays reproducible from --seed.
    import random as _random
    _rank = int(os.environ.get("RANK", 0))
    torch.manual_seed(args.seed + _rank)
    torch.cuda.manual_seed_all(args.seed + _rank)
    np.random.seed(args.seed + _rank)
    _random.seed(args.seed + _rank)

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    lead = rank == 0

    tr = GenomeWindowDataset(args.windows, args.manifest, args.fasta,
                              split="train", require_local=False,
                              max_tracks=args.max_tracks, track_seed=args.track_seed,
                              replicate_to=args.replicate_to, bin_size=args.bin_size)
    va = GenomeWindowDataset(args.windows, args.manifest, args.fasta,
                              split="val", require_local=False,
                              max_tracks=args.max_tracks, track_seed=args.track_seed,
                              replicate_to=args.replicate_to, bin_size=args.bin_size)
    T = tr.n_tracks()
    if T == 0:
        # Distinguish "data not produced yet" from "manifest paths do not resolve from HERE".
        # A manifest that stores RELATIVE paths silently resolves against the wrong root when the
        # launcher runs from a different directory, and every present file then looks absent. This
        # is why DATA_SCHEMA.md asks for absolute paths in the manifest.
        import json as _json
        _man = _json.loads(Path(args.manifest).read_text())
        _rel = [t["path"] for t in _man if t.get("path") and not t["path"].startswith("/")]
        _msg = [f"0 of {len(_man)} tracks in {Path(args.manifest).name} resolved to a local file"]
        if _rel:
            _msg.append(f"  {len(_rel)}/{len(_man)} paths are RELATIVE, e.g. {_rel[0]!r}")
            _msg.append(f"  cwd is {Path.cwd()} -- absolutise the manifest, do not rely on cwd")
        else:
            _msg.append("  paths are absolute but absent -- data really is not produced yet")
        raise SystemExit("\n".join(_msg))
    # train and val must select the SAME channels in the SAME order, or every val number is
    # comparing channel k of one set against channel k of another. Same seed is not enough to
    # trust silently -- assert it.
    if [t["track_id"] for t in tr.tracks] != [t["track_id"] for t in va.tracks]:
        raise SystemExit("train/val track selection differs -- channel k means different assays")
    if lead:
        from collections import Counter
        print(f"tracks={T} (manifest {args.manifest.name}; {len(tr.missing)} pending)  "
              f"train={len(tr):,}  val={len(va):,}", flush=True)
        print(f"  assay mix: {dict(Counter(t.get('assay','?') for t in tr.tracks))}", flush=True)
        srcs = {t.get("source") for t in tr.tracks}
        if len(srcs) > 1:
            print(f"  🚨 MIXED PROVENANCE {srcs}: the convention check FAILED between these "
                  f"pipelines. A model can learn provenance instead of biology.", flush=True)

    # per-track scaling: divide by the track's own mean so no assay dominates the loss
    # `t.get("track_mean") or 1.0` would map a legitimate 0.0 to 1.0 and hide a dead track;
    # a 0.0 that survived into the divisor gives inf scaling. Treat None and <=0 as "no scale"
    # and SAY how many, rather than silently papering over them.
    _raw = [t.get("track_mean") for t in tr.tracks]
    _bad = sum(1 for m in _raw if m is None or m <= 0)
    means = np.array([m if (m is not None and m > 0) else 1.0 for m in _raw], dtype=np.float32)
    means = np.clip(means, 1e-3, None)
    if lead:
        print(f"  per-track scaling from track_mean: min={means.min():.3f} "
              f"max={means.max():.3f} (1.0 = not yet computed)", flush=True)
        if _bad:
            print(f"  ⚠️ {_bad}/{len(_raw)} tracks had no usable track_mean (None or <=0) "
                  f"and are UNSCALED", flush=True)
        if tr.replicated_from is not None:
            print(f"  🚨 CAPACITY PROBE: {T} channels REPLICATED from {tr.replicated_from} "
                  f"real tracks. Memory/throughput are faithful; the MODEL IS NOT TRAINABLE "
                  f"(duplicated channels = duplicated loss). Never report accuracy from this.",
                  flush=True)
    scale = torch.tensor(1.0 / means, device=dev)

    # AG's target transform (paper p.32) is TWO steps; `scale` above is only the first.
    # The sqrt clip applies to every track; the **0.75 power to RNA-seq columns only.
    # SUBSTRING, not startswith. ENCODE names the assay "polyA plus RNA-seq", so startswith
    # matched 0 of 22 human RNA tracks and the AG squashing would have silently skipped them.
    # Harmless at --target-clip none (the only setting we use), fatal to the clip's meaning
    # otherwise. Line 517's gene-tissue selector already used a substring test -- these two
    # disagreed about what an RNA track is.
    _is_rna = ["RNA" in str(t.get("assay", "")).upper() for t in tr.tracks]
    rna_mask = torch.tensor(_is_rna, dtype=torch.bool, device=dev)
    tgt_clip = (args.target_clip == "ag")
    if lead:
        print(f"  target scaling: divide-by-track_mean"
              f"{' + AG sqrt clip' if tgt_clip else ' ONLY (not AG-faithful)'}"
              f"; RNA columns for **0.75: {int(rna_mask.sum())}/{len(_is_rna)}", flush=True)

    def scale_targets(y):
        """y is already multiplied by `scale`; apply AG's step 2 if enabled."""
        if not tgt_clip:
            return y
        return PL.targets_scaling(y, rna_mask=rna_mask, clip=True, squash_rna=True)

    model, head = build_model(T, args.num_organisms, args.checkpoint,
                              args.grad_ckpt, dev)

    if args.head_init == "ag":
        init_head_from_ag(head, tr.tracks, args.checkpoint,
                          seed=(args.seed if args.head_init_seed < 0 else args.head_init_seed),
                          verbose=lead)

    # ---- E-SERIES FREEZING -------------------------------------------------------------
    # Groups are TOP-LEVEL AlphaGenome children, measured not assumed:
    #   encoder 89,972,992 | tower 191,801,088 | decoder 111,501,447
    #   embedder_1bp 5,905,920 | embedder_128bp 4,733,952 | embedder_pair 512 (already frozen)
    # The embedders sit between tower and decoder outputs and the heads; they are BACKBONE, so
    # they follow the decoder group rather than the head.
    BACKBONE = {"head":         ["encoder", "tower", "decoder", "embedder_1bp", "embedder_128bp"],
                "head+decoder": ["encoder", "tower"],
                "all":          []}
    frozen_groups = BACKBONE[args.trainable]
    for gname in frozen_groups:
        mod = getattr(model, gname, None)
        if mod is None:
            raise SystemExit(f"--trainable {args.trainable}: model has no module '{gname}'")
        for prm in mod.parameters():
            prm.requires_grad_(False)
    # organism_embed lives in FOUR places (root + embedder_1bp/128bp/pair); re-enable them all
    # after the sweep above, or freezing the embedders would silently take it down with them.
    n_oe = 0
    if args.organism_embed_trainable:
        # ⚠️ SKIP embedder_pair. build_model freezes it ON PURPOSE: it feeds the contact-maps
        # head, which we delete, so it NEVER RECEIVES A GRADIENT. Walking every module and
        # re-enabling `organism_embed` silently undid that freeze. At world=1 the param merely
        # sits in the optimizer and never updates (Adam creates no state for it -- which is how
        # the resume smoke test surfaced this: 6/10 restored instead of 10/10). At world>1 it is
        # fatal: DDP raises "Expected to have finished reduction" on params that get no grad.
        _pair = getattr(model, "embedder_pair", None)
        _pair_ids = {id(q) for q in _pair.parameters()} if _pair is not None else set()
        for _m in model.modules():
            _e = getattr(_m, "organism_embed", None)
            if isinstance(_e, torch.nn.Embedding):
                for prm in _e.parameters():
                    if id(prm) in _pair_ids:
                        continue
                    prm.requires_grad_(True); n_oe += prm.numel()
    if lead:
        _tr = sum(p.numel() for p in list(model.parameters()) + list(head.parameters())
                  if p.requires_grad)
        _to = sum(p.numel() for p in list(model.parameters()) + list(head.parameters()))
        print(f"  --trainable={args.trainable}: froze {frozen_groups or '(nothing)'}; "
              f"organism_embed trainable={bool(args.organism_embed_trainable)} ({n_oe} params)",
              flush=True)
        print(f"  TRAINABLE {_tr:,} / {_to:,} params ({100*_tr/_to:.2f}%)", flush=True)
    # ⚠️ NAME the trainable params. Adam's state_dict keys state by INDEX into param_groups, so
    # if the trainable set changes between a save and a resume -- which is EXACTLY what E4's
    # staged schedule does (P1 head -> P2 +decoder -> P3 +tower) -- indices shift and a naive
    # load silently assigns one tensor's moments to a different tensor. Keying by name makes
    # that impossible: params present in both get their moments back, newly-unfrozen ones start
    # fresh, and a mismatch is visible instead of silent.
    def named_trainable():
        for _n, _p in model.named_parameters():
            if _p.requires_grad:
                yield f"model.{_n}", _p
        for _n, _p in head.named_parameters():
            if _p.requires_grad:
                yield f"head.{_n}", _p

    _named = list(named_trainable())
    params = [p for _, p in _named]
    param_names = [n for n, _ in _named]
    opt = torch.optim.Adam(params, lr=args.lr)
    if ddp:
        model = DDP(model, device_ids=[local])
        head = DDP(head, device_ids=[local])

    def loader(ds, shuffle):
        from torch.utils.data import DataLoader, DistributedSampler
        samp = DistributedSampler(ds, shuffle=shuffle) if ddp else None
        return DataLoader(ds, batch_size=args.batch_size, shuffle=(shuffle and samp is None),
                          sampler=samp, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.workers > 0)

    # ---- gene-level ACROSS-TISSUE loss setup -------------------------------------------
    # AlphaGenome carries a Decima-inspired auxiliary term (paper p.32) whose whole job is to
    # get the distribution across tissues right per gene, at overall weight 0.1. We trained
    # without it and measured across-tissue r = 0.051, so it is the first thing to restore.
    def compute_loss(pred_head, y, widx=None):
        """The full objective. Returns (loss, parts dict) -- parts are logged, not optimised."""
        rate = PL.rate_from_head(pred_head, args.rate_param)
        # ONE denominator for every term, so the paper's weights (5.0, 0.1) keep the meaning the
        # paper gives them. Dividing everything by the same constant preserves the ratios and
        # only rescales the learning rate.
        denom = rate.shape[0] * rate.shape[1] * rate.shape[2]
        # 🔁 SWAP YOUR OWN LOSS IN HERE. Contract: take (rate, target) with rate >= 0 and shape
        #    (B, L, T), return (scalar_loss, {name: float} for logging). Everything downstream --
        #    logging, validation, checkpointing -- goes through this one function.
        total, pois, mult = PL.ag_loss(rate, y, args.n_segments,
                                       args.multinomial_weight,
                                       args.poisson_weight, denom)
        parts = {"poisson": float(pois), "multinomial": float(mult)}
        # COVERAGE RATIO. The multinomial cannot see overall scale, so this is the only readout
        # that says whether the Poisson anchor is actually doing its job. 1.0 = calibrated.
        with torch.no_grad():
            parts["cov_ratio"] = float(rate.sum() / y.sum().clamp_min(1e-6))
        return total, parts

    # ---- WHICH VAL LOCI ---------------------------------------------------------------------
    # `prefix` reproduces the historical path exactly (plain loader, break at --val-batches).
    # The other modes hand validate() an explicit, deterministic list of window indices.
    _n_val_w = max(1, args.val_batches * args.batch_size)
    val_order = None
    if args.val_select != "prefix":
        _tot = len(va)
        if _tot <= _n_val_w:
            val_order = list(range(_tot))
        else:
            # evenly spaced across the WHOLE split -- deterministic, no RNG, and identical for
            # any two arms that share a window index (which is what makes arms comparable).
            val_order = [int(round(i * (_tot - 1) / (_n_val_w - 1))) for i in range(_n_val_w)] \
                        if _n_val_w > 1 else [_tot // 2]
        if lead:
            print(f"  --val-select {args.val_select}: {len(val_order)} val windows of {len(va)} "
                  f"({100*len(val_order)/max(len(va),1):.2f}% of the split), "
                  f"{'rotating each validation' if args.val_select == 'rotate' else 'fixed'}",
                  flush=True)
    elif lead:
        print(f"  ⚠️ --val-select prefix: scoring the FIRST {_n_val_w} of {len(va)} val windows "
              f"({100*_n_val_w/max(len(va),1):.2f}% of the split), THE SAME ONES EVERY "
              f"VALIDATION. Averaging validations does not widen this. See --help.", flush=True)

    dl_tr, dl_va = loader(tr, True), loader(va, False)
    core = model.module if ddp else model
    hd = head.module if ddp else head

    crop = tr.crop
    out_bp = tr.output_bp

    def forward(x, org):
        """Predict, then CROP to the output window.

        AlphaGenome's 1 bp head spans the ENTIRE input -- measured: a 1,048,576 bp input gives
        embeddings_1bp of (1, 1536, 1048576), with no internal cropping. Our targets cover only
        the central `output_bp`, because the outer `crop` bp on each side exist to give edge
        positions two-sided context. Comparing the full prediction against the cropped target is
        a shape error (8192 vs 16384); worse, training on the uncropped output would fit edge
        positions that only ever see context from one side.
        """
        # Ask the TRUNK for only the resolution we train on. This is not a micro-optimisation:
        # when 1 bp is not requested the model skips self.decoder entirely -- the upsampling
        # stack that produces (B, 768, 1,048,576) and then (B, 1536, 1,048,576) embeddings --
        # and frees the encoder's skip-connection intermediates. That is trunk memory, which
        # binning the TARGETS alone could never reach.
        res = args.bin_size
        out = core(x, org, embeddings_only=True, channels_last=False, resolutions=(res,))
        key = "embeddings_1bp" if res == 1 else "embeddings_128bp"
        # Only the resolution we train on: previously BOTH were passed to the head and the
        # 128 bp prediction was computed and discarded every step.
        pred = hd({res: out[key]}, org, return_scaled=True, channels_last=True)[res]
        n_bins = out_bp // args.bin_size
        crop_bins = crop // args.bin_size
        if pred.shape[1] != n_bins:
            if pred.shape[1] != n_bins + 2 * crop_bins:
                raise RuntimeError(
                    f"unexpected prediction length {pred.shape[1]}; expected "
                    f"{n_bins + 2*crop_bins} (input) or {n_bins} (already cropped) "
                    f"at bin_size={args.bin_size}")
            pred = pred[:, crop_bins:crop_bins + n_bins, :]
        return pred

    def _batched(sel):
        """Yield (x, y) batches for an explicit list of window indices, in order."""
        from torch.utils.data import default_collate
        for j in range(0, len(sel), args.batch_size):
            yield default_collate([va[k] for k in sel[j:j + args.batch_size]])

    @torch.no_grad()
    def validate():
        """Report per-track correlation AND the baselines that make it interpretable."""
        core.eval(); hd.eval()
        ys, ps = [], []
        if val_order is None:
            _it = ((i, b) for i, b in enumerate(dl_va))
        else:
            # rotate: advance the block by one window per validation so successive validations
            # cover DIFFERENT loci; an average over the last N validations then rests on ~N times
            # as many distinct loci as `spread` at the same per-validation cost.
            if args.val_select == "spread":
                _sel = list(val_order)                       # same loci every validation
            else:                                            # rotate
                _off = validate.n_calls
                _sel = [(w + _off) % len(va) for w in val_order]
            _it = enumerate(_batched(_sel))
        validate.n_calls += 1
        for i, (x, y) in _it:
            if val_order is None and i >= args.val_batches:
                break
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            org = torch.full((x.shape[0],), args.organism_index, dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pl = forward(x, org)
            ys.append(scale_targets(y * scale).float().cpu())
            ps.append(PL.rate_from_head(pl.float(), args.rate_param).cpu())
        core.train(); hd.train()
        if not ys:
            return {}
        Y = torch.cat(ys); P = torch.cat(ps)                       # (N, L, T)
        Yf = Y.permute(2, 0, 1).reshape(T, -1)
        Pf = P.permute(2, 0, 1).reshape(T, -1)
        def pear(a, b):
            a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
            return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8))
        r_model = pear(Pf, Yf)
        # ACROSS-TRACK r: at each position, is the profile ACROSS the 48 tracks right? This is
        # the specificity metric a profile-memoriser fails, and the one the loss work is aimed
        # at -- within-window r is dominated by structure every track shares. Subsampled
        # because it is a per-position reduction over millions of positions.
        with torch.no_grad():
            Pp = P.reshape(-1, T); Yp = Y.reshape(-1, T)
            live = Yp.std(1) > 1e-6
            Pp, Yp = Pp[live], Yp[live]
            if Pp.shape[0] > 200_000:
                sel = torch.randperm(Pp.shape[0])[:200_000]
                Pp, Yp = Pp[sel], Yp[sel]
            # 🚨 `r_track` IS MISLABELLED AND IS **NOT** AN ACROSS-TRACK METRIC.
            # pear() correlates along dim=1 and returns one value PER ROW. Pp is (positions, T),
            # so Pp.T is (T, positions) and this returns T values -- one per TRACK, correlating
            # that track's profile across positions. That is the SAME STATISTIC as `r_model`
            # above, differing ONLY by the live-filter and the 200k subsample. Verified
            # synthetically: pear(Pp.T, Yp.T) and pear(Pf, Yf) return identical value sets on the
            # same positions.
            # ⚠️ KEPT UNCHANGED ON PURPOSE so F3, F4 and the E-ladder histories stay comparable
            # to each other -- do not "fix" it in place or every prior number silently shifts.
            # The correctly-computed metric is `r_across_median`, added below. Measured gap on a
            # real checkpoint: r_track 0.72 vs true across-track 0.39.
            r_track = pear(Pp.T.contiguous(), Yp.T.contiguous()) if Pp.shape[0] else torch.zeros(1)
            # TRUE ACROSS-TRACK: at each POSITION, is the profile across the T tracks right?
            # One value per position, not per track. This is the specificity metric a
            # profile-memoriser fails, and the one the loss work was actually aimed at.
            if Pp.shape[0]:
                _a = Pp - Pp.mean(1, keepdim=True)
                _b = Yp - Yp.mean(1, keepdim=True)
                r_across = ((_a * _b).sum(1) /
                            (_a.norm(dim=1) * _b.norm(dim=1)).clamp_min(1e-8))
            else:
                r_across = torch.zeros(1)
            # coverage calibration -- the multinomial cannot see scale, so track it explicitly
            cov = float(P.sum() / Y.sum().clamp_min(1e-6))
            # ⚠️ COMPARABILITY ACROSS --target-clip ARMS. With the clip on, P and Y both live in
            # CLIPPED space, so `cov` above measures agreement there and is NOT comparable to a
            # run without the clip. Invert both back to original units for a number that is.
            # With the clip off this is the identity, so the two agree by construction.
            if tgt_clip:
                P_o = PL.predictions_scaling(P, rna_mask=rna_mask.cpu(), clip=True, squash_rna=True)
                Y_o = PL.predictions_scaling(Y, rna_mask=rna_mask.cpu(), clip=True, squash_rna=True)
                cov_orig = float(P_o.sum() / Y_o.sum().clamp_min(1e-6))
            else:
                cov_orig = cov
        # BASELINE: predict the mean profile of this validation batch for every window.
        # A model that only learns the average accessibility landscape scores well on
        # within-window correlation; if it cannot beat this, it has learned nothing.
        # LEAVE-ONE-OUT. Averaging ALL validation windows including the one being predicted
        # leaks 1/N of the answer into the control, and the leak scales with 1/N: with the
        # default val_batches=40 it reads ~0.15, with 120 windows ~0.09, on the SAME data.
        # That made the control a sample-size readout, not a baseline. Built from the other
        # windows it sits at ~0, which is correct -- other loci say nothing about this one.
        n = Y.shape[0]
        mean_loo = (Y.sum(0, keepdim=True) - Y) / max(n - 1, 1)
        r_mean = pear(mean_loo.permute(2, 0, 1).reshape(T, -1), Yf)
        mean_self = Y.mean(0, keepdim=True).expand_as(Y)
        r_mean_self = pear(mean_self.permute(2, 0, 1).reshape(T, -1), Yf)
        return {"r_model_median": float(r_model.median()),
                "r_track_median": float(r_track.median()),
                "r_across_median": float(r_across.median()),
                "n_positions_scored": int(Pp.shape[0]),
                "cov_ratio": cov, "cov_ratio_orig_units": cov_orig,
                "r_meanprofile_median": float(r_mean.median()),
                "r_meanprofile_median_selfincluded": float(r_mean_self.median()),
                "n_val_windows": int(n),
                "r_model_per_track": [round(float(v), 4) for v in r_model],
                "beats_baseline": int((r_model > r_mean).sum())}

    validate.n_calls = 0
    args.out.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path, step, metrics=None):
        """Write weights atomically. Saved AT EVERY VALIDATION, not only at the end.

        A 10,000-step run is ~7 hours; saving once at the end means a wall-clock timeout or a
        node fault at step 9,000 leaves history.json full of metrics and no model to show for
        them. The temp-then-rename keeps a killed job from leaving a truncated .pt that looks
        loadable until it is loaded."""
        tmp = path.with_suffix(".pt.tmp")
        # OPTIMIZER STATE IS PART OF THE MODEL'S TRAINING TRAJECTORY, not an optional extra.
        # Without it a "resume" hands Adam ZEROED moments; after thousands of steps of
        # accumulated second-moment estimates that abruptly changes the effective step size and
        # injects a transient -- exactly where a convergence check would be looking. Its absence
        # is why F3 had to be retrained from scratch at 6,000 steps (~36 GPU-h) instead of
        # extended from 2,000.
        _osd = opt.state_dict()
        _opt_by_name = {param_names[i]: v for i, v in _osd["state"].items()
                        if i < len(param_names)}
        torch.save({"model": core.state_dict(), "head": hd.state_dict(),
                    "opt_state_by_name": _opt_by_name,
                    "opt_param_groups": _osd["param_groups"],
                    "step": step, "metrics": metrics,
                    "args": vars(args) | {"out": str(args.out),
                                          "checkpoint": str(args.checkpoint),
                                          "windows": str(args.windows),
                                          "manifest": str(args.manifest),
                                          "fasta": str(args.fasta)},
                    "tracks": tr.track_table()}, tmp)
        tmp.replace(path)

    torch.cuda.reset_peak_memory_stats()
    if lead:
        # Record the allocator config IN THE LOG. A capacity run measured 1,100 channels
        # as the wall, and whether that number is the real ceiling or a pessimistic one depends
        # on whether expandable_segments actually reached the workers -- which the logs could
        # not answer after the fact. Print it so the question never has to be inferred again.
        print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}"
              f"  bin_size={args.bin_size}bp  n_bins={tr.n_bins}", flush=True)
    hist, step, t0 = [], 0, time.time()
    best_r = float("-inf")
    if args.resume is not None:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        mm, mu = core.load_state_dict(rck["model"], strict=False)
        hm, hu = hd.load_state_dict(rck["head"], strict=False)
        if mm or mu or hm or hu:
            raise SystemExit(f"--resume: checkpoint does not match the model (model "
                             f"missing={len(mm)} unexpected={len(mu)}, head missing={len(hm)} "
                             f"unexpected={len(hu)}). Resuming a partial load would train a "
                             f"different model than the one that produced the metrics.")
        saved = rck.get("opt_state_by_name")
        if saved is None:
            raise SystemExit("--resume: this checkpoint predates optimizer-state saving. "
                             "Resuming from it would zero Adam's moments and inject a "
                             "transient; retrain from scratch instead, or accept that and pass "
                             "--resume-allow-no-opt (not implemented on purpose).")
        new_state, restored, fresh = {}, 0, []
        for i, n in enumerate(param_names):
            if n in saved:
                new_state[i] = saved[n]; restored += 1
            else:
                fresh.append(n)
        opt.load_state_dict({"state": new_state,
                             "param_groups": opt.state_dict()["param_groups"]})
        step = int(rck.get("step", 0))
        hp = args.out / "history.json"
        if hp.exists():
            try:
                hist = [h for h in json.loads(hp.read_text()) if h.get("step", 0) <= step]
            except Exception:
                hist = []
        if lead:
            print(f"  RESUMED from {args.resume} at step {step}: "
                  f"{restored}/{len(param_names)} params got their Adam moments back, "
                  f"{len(fresh)} start fresh", flush=True)
            if fresh:
                print(f"    fresh (were frozen in the saved run): {fresh[:4]}"
                      f"{' ...' if len(fresh) > 4 else ''}", flush=True)
            if step >= args.steps:
                print(f"  ⚠️ resume step {step} >= --steps {args.steps}: nothing to do. "
                      f"Raise --steps to extend.", flush=True)
    if lead:
        print("step | loss | val r(model) | val r(mean-profile) | tracks beating baseline",
              flush=True)
    while step < args.steps:
        for batch in dl_tr:
            if step >= args.steps:
                break
            widx = None
            if tr.return_index:
                x, y, widx = batch
            else:
                x, y = batch
            x = x.to(dev, non_blocking=True)
            y = scale_targets(y.to(dev, non_blocking=True) * scale)
            org = torch.full((x.shape[0],), args.organism_index, dtype=torch.long, device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pl = forward(x, org).float()
                loss, loss_parts = compute_loss(pl, y, widx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            # Early steps are printed individually: a run that dies at step 2 and a run that
            # dies before step 1 fail very differently, and with logging only every 250 steps
            # both look identical -- a job that produced no output at all.
            if lead and (step <= 3 or step % args.log_every == 0):
                peak = torch.cuda.max_memory_allocated() / 1024**3
                resv = torch.cuda.max_memory_reserved() / 1024**3
                # Log the TERMS, not just the total. The multinomial dominates the total by
                # orders of magnitude, so a moving total says nothing about whether the
                # magnitude anchor or the tissue term is doing anything.
                pstr = "  ".join(f"{k}={v:+.4e}" for k, v in loss_parts.items())
                print(f"  step {step:>5}  loss={loss.item():.4f}  {pstr}  "
                      f"{(time.time()-t0)/step:.2f}s/step  "
                      f"PEAK_GPU_GIB={peak:.2f} reserved={resv:.2f}", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                m = validate()
                if lead and m:
                    print(f"{step:>5} | {loss.item():.4f} | r_win={m['r_model_median']:.4f} | "
                          f"r_track={m['r_track_median']:.4f} | r_across={m['r_across_median']:.4f} | "
                          f"cov={m['cov_ratio']:.3f} | "
                          f"base={m['r_meanprofile_median']:+.4f} | "
                          f"{m['beats_baseline']}/{T}", flush=True)
                    hist.append({"step": step, "loss": float(loss.item()),
                                 "loss_parts": loss_parts, "rate_param": args.rate_param, "target_clip": args.target_clip, "trainable": args.trainable,
                                  **m})
                    (args.out / "history.json").write_text(json.dumps(hist, indent=1) + "\n")
                    save_ckpt(args.out / "last.pt", step, m)
                    # the final step is not necessarily the best one; keep both so a late
                    # overfit cannot silently become the only model we have
                    if m["r_model_median"] > best_r:
                        best_r = m["r_model_median"]
                        save_ckpt(args.out / "best.pt", step, m)
                        print(f"        new best r_model={best_r:.4f} at step {step} -> best.pt",
                              flush=True)
    if lead:
        save_ckpt(args.out / "last.pt", step, hist[-1] if hist else None)
        print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")
        print(f"FINAL_PEAK_GPU_GIB={torch.cuda.max_memory_allocated()/1024**3:.2f} "
              f"tracks={T} input_bp={tr.input_bp} world={world}", flush=True)
        print("TRAIN_DONE")
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
