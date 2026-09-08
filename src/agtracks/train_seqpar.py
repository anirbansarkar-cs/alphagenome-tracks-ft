#!/usr/bin/env python3
"""Train at a full 1 Mb window with SEQUENCE PARALLELISM, when DDP runs out of track capacity.

WHY THIS EXISTS, AND WHEN NOT TO USE IT.
  DDP replicates the model, so each rank pays weights + the FULL activation for the FULL sequence.
  Adding GPUs buys throughput and **can never raise the track ceiling**. Measured on 93 GiB cards
  at 1 bp output, DDP tops out around **1,100 output channels no matter how many GPUs you add**.

  Sequence parallelism shards the SEQUENCE instead of the batch, so per-GPU activation and head
  memory fall roughly with world size, and the track ceiling scales with it:

      world = 1      ~1,100 channels     (this is just the single-GPU limit)
      world = 2      ~4,000 channels
      world = 4      8,000+ channels     (8,000 measured to fit; 12,000 OOMs, not bisected)

  Per-track cost halves exactly from world 2 to 4: 14.0 -> 7.1 MB.

⚠️ IT BUYS CAPACITY, NOT SPEED. Throughput scales poorly -- world 2 -> 4 measured **1.23x**, not
   2x. If your tracks already fit under DDP, use DDP: it is simpler and faster per step. Reach for
   this only when the track count is the binding constraint.

TWO DESIGN POINTS that are easy to get wrong if you reimplement it:
  1. THE SAMPLER IS NOT A DistributedSampler. Under DDP each rank takes a DIFFERENT window. Under
     sequence parallelism every rank takes the SAME window and a different slice of its POSITIONS.
     Ranks walk an identical, identically-seeded window order, and the dataset serves each rank
     only its positional shard via `shard_rank` / `shard_world`.
  2. THE TARGET IS EXPANDED IN CHUNKS. A wide 1 bp target is gigabytes per rank in fp32;
     materialising it whole doubles the head's own output cost for nothing. The loss accumulates
     over track chunks, which is mathematically identical because a mean over all channels equals
     the size-weighted mean of per-chunk means.

Predictions are clamped before the implicit exp(): with a randomly-initialised head the raw values
overflow fp32 and give inf loss with NaN gradients.

CORRECTNESS. The sharded and unsharded forward passes were compared directly: loss exact to
1.3e-6 and gradient cosine 0.9996, with the worst case 2.7e-2 relative on tower attention biases
and flat across an 8x overlap sweep, so it is not a seam artefact. `--no-seqpar` runs the identical
data and schedule on one GPU so the loss trajectories can be compared.
⚠️ Do NOT validate a long sharded run by weight identity. Agreement degrades superlinearly because
   Adam normalises by gradient magnitude -- weight 1-cos went 2e-09 at 150 steps to 7.7e-06 at 600.
   Compare validation METRICS.

REQUIREMENT ON GEOMETRY: the window index's `output_bp` must be divisible by the world size,
because each rank takes an equal slice of the output positions. The dataset raises if it is not.

Launch (4 GPUs, sequence-parallel):
    torchrun --nproc_per_node=4 -m agtracks.train_seqpar \
        --windows W.json --manifest M.json --fasta genome.fa \
        --checkpoint AG.safetensors --organism-index 0 \
        --steps 2000 --out runs/sp

Launch (single-GPU control, to compare loss trajectories against the sharded run):
    python -m agtracks.train_seqpar --no-seqpar \
        --windows W.json --manifest M.json --fasta genome.fa \
        --checkpoint AG.safetensors --organism-index 0 \
        --steps 2000 --out runs/control
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

LOGIT_CLAMP = 15.0
TRACK_CHUNK = 512


def chunked_poisson(pred, y_uniq, idx, clamp=LOGIT_CLAMP):
    """Mean Poisson NLL of `pred` (B,S,T) against y_uniq (B,S,U) fanned out through idx (T,).

    Accumulated over track chunks so the expanded target never exists in full. The chunk means
    are combined size-weighted, which equals the mean over all T channels exactly.
    """
    T = pred.shape[-1]
    total, seen = 0.0, 0
    for a in range(0, T, TRACK_CHUNK):
        b = min(a + TRACK_CHUNK, T)
        tgt = y_uniq.index_select(-1, idx[a:b])
        p = pred[..., a:b].float().clamp(-clamp, clamp)
        total = total + F.poisson_nll_loss(p, tgt, log_input=True, reduction="mean") * (b - a)
        seen += b - a
    return total / seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    required=True)
    ap.add_argument("--windows", type=Path,
                    required=True)
    ap.add_argument("--fasta", type=Path,
                    required=True)
    ap.add_argument("--checkpoint", type=Path,
                    required=True)
    ap.add_argument("--organism-index", type=int, default=3)
    ap.add_argument("--num-organisms", type=int, default=4)
    ap.add_argument("--replicate-to", type=int, default=None,
                    help="BENCHMARKING ONLY. Fan the output channels out to this many by "
                         "REPEATING tracks, to measure a capacity ceiling without needing "
                         "that many real bigwigs. Default None = train on your real tracks, "
                         "which is what you want. Setting it trains on DUPLICATED targets, "
                         "so the resulting model is not meaningful.")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="subset the manifest DOWN to this many real tracks first; needed for "
                         "the trajectory control, whose single-GPU arm must fit on one card")
    ap.add_argument("--seq-len", type=int, default=1048576)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="matches agtracks.train. At 1e-5 the model did not move at all "
                         "(r_model 0.0193 -> 0.0002 over 200 steps), which would have made the "
                         "quality comparison vacuous: two arms that both learn nothing agree "
                         "trivially.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-seqpar", action="store_true",
                    help="single-GPU control: normal model path, no sharding")
    ap.add_argument("--ddp", action="store_true",
                    help="DDP mode: every rank takes a DIFFERENT window and a full model copy, "
                         "gradients averaged. Identical loss and data handling to the seq-par "
                         "path, so step times are comparable arm-to-arm.")
    ap.add_argument("--overlap-high", type=int, default=1024)
    ap.add_argument("--head-ncl", action="store_true",
                    help="ask the head for NCL (B,T,S) instead of NLC. The package documents NCL "
                         "as 'for training efficiency (0 transposes)'; at 1 Mb the NLC transpose "
                         "moves a very large tensor twice (forward and backward).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader prefetch workers. 0 = read inline (the old behaviour). "
                         "Profiling showed bigwig reads were 86%% of step time (11.0s of 12.8s) "
                         "with the GPU idle throughout, so overlapping them is worth far more "
                         "than any hardware choice.")
    ap.add_argument("--profile", action="store_true",
                    help="break the step into phases (data / H2D / trunk / head / loss / "
                         "backward / all-reduce / optimizer). Adds cuda.synchronize() between "
                         "phases, which slows the step slightly but is the only way to ATTRIBUTE "
                         "it -- without it a slow dataloader is indistinguishable from slow compute.")
    ap.add_argument("--val-every", type=int, default=0,
                    help="0 disables. Validation is the ONLY correctness criterion that stays "
                         "meaningful over long runs: weight identity provably decays (1-cos "
                         "grew 3,800x from 150 to 600 steps) and would decay for a CORRECT "
                         "implementation too, because training is chaotic.")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--save-final", type=Path, default=None,
                    help="dump final parameters, so sharded and single-GPU runs can be compared "
                         "at the WEIGHT level. Matching losses do not prove matching weights: "
                         "two runs can ride the same loss curve while their parameters drift.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.ddp:
        args.no_seqpar = True                 # DDP uses the unsharded forward path
    seqpar = not args.no_seqpar
    if seqpar or args.ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1
    dev = f"cuda:{rank}"
    lead = rank == 0

    from agtracks.dataset import GenomeWindowDataset
    # Reuse the existing builder rather than re-deriving it. It carries two things this script
    # would otherwise get wrong: AlphaGenome's constructor leaves splice track counts UNDEFINED
    # for num_organisms > 2 (UnboundLocalError before we can even delete the splice heads), so
    # it builds with 2 and widens the organism embeddings; and it REFUSES random init, because
    # RMSBatchNorm never updates running_var -- an uninitialised model would train with no
    # normalisation at all and report nothing.
    from agtracks.train import build_model

    ds = GenomeWindowDataset(
        args.windows, args.manifest, args.fasta, split="train", require_local=False,
        bin_size=1, shard_rank=rank if seqpar else 0,
        shard_world=world if seqpar else 1,
        max_tracks=args.max_tracks)
    # The WINDOW FILE determines the sequence length, not a CLI flag. --seq-len only ever fed
    # `original_length` and the log line, so passing a different value silently trained at the
    # window file's length while telling sequence parallelism to trim to another -- which is
    # exactly how arm B died (target 524,288 vs embeddings 131,072) and why arm A's memory
    # looked anomalous (it was 1 Mb all along, not the 262 kb the flag claimed).
    if args.seq_len != ds.input_bp:
        raise SystemExit(
            f"--seq-len {args.seq_len:,} disagrees with the window index ({ds.input_bp:,} bp "
            f"in {Path(args.windows).name}). The window file is authoritative; either pass "
            f"--seq-len {ds.input_bp} or build a window index at the length you want.")
    ds_va = GenomeWindowDataset(
        args.windows, args.manifest, args.fasta, split="val", require_local=False,
        bin_size=1, shard_rank=rank if seqpar else 0,
        shard_world=world if seqpar else 1,
        max_tracks=args.max_tracks) if args.val_every else None
    n_uniq = len(ds.tracks)
    T = args.replicate_to if args.replicate_to else n_uniq
    if T < n_uniq:
        raise SystemExit(
            f"--replicate-to {T} < {n_uniq} real tracks. Replication only EXPANDS; to train on "
            f"fewer channels pass --max-tracks {T} as well (it subsets the manifest first).")
    idx = torch.tensor([i % n_uniq for i in range(T)], dtype=torch.long, device=dev)
    # Per-track scaling. None/<=0 are treated as "no scale" rather than silently becoming 1.0
    # via truthiness (a legitimate 0.0 would also be a division blow-up).
    _raw = [t.get("track_mean") for t in ds.tracks]
    _bad = sum(1 for m in _raw if m is None or m <= 0)
    _means = np.array([m if (m is not None and m > 0) else 1.0 for m in _raw], dtype=np.float32)
    scale = torch.tensor(1.0 / np.clip(_means, 1e-3, None), device=dev)

    if lead:
        print(f"seqpar={seqpar} world={world} seq_len={args.seq_len:,} "
              f"per_rank_bp={args.seq_len // world:,}", flush=True)
        print(f"channels={T} fanned out from {n_uniq} real tracks; "
              f"resolutions=(1,128) for ALL channels", flush=True)
        print(f"  lr={args.lr}  per-track scaling {float(_means.min()):.3f}.."
              f"{float(_means.max()):.3f}" + (f"  ({_bad} unscaled)" if _bad else ""), flush=True)
        print(f"PYTORCH_CUDA_ALLOC_CONF="
              f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}", flush=True)

    torch.manual_seed(args.seed)
    model, head = build_model(num_tracks=T, num_organisms=args.num_organisms,
                              checkpoint=args.checkpoint, grad_ckpt=True, device=dev)

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    sp = None
    if seqpar:
        from alphagenome_pytorch.sequence_parallel import create_sequence_parallel_strategy
        if args.overlap_high % 128:
            raise SystemExit("overlap_high must be a multiple of 128")
        sp = create_sequence_parallel_strategy(overlap_highres=args.overlap_high,
                                               overlap_lowres=args.overlap_high // 128)

    order = np.random.default_rng(args.seed).permutation(len(ds))

    class _Windows(torch.utils.data.Dataset):
        """Serves this rank's windows in the fixed order, so workers can prefetch them.

        The order is identical to the inline version: seq-par ranks share a window, DDP ranks
        take different ones. Only WHEN the read happens changes, never WHICH window.
        """
        def __len__(self):
            return args.steps

        def __getitem__(self, i):
            wi = (i * world + rank) if args.ddp else i
            return ds[int(order[wi % len(order)])]

    loader = None
    if args.workers > 0:
        loader = torch.utils.data.DataLoader(
            _Windows(), batch_size=None, shuffle=False, num_workers=args.workers,
            pin_memory=True, prefetch_factor=2, persistent_workers=True)
        it = iter(loader)
    org = torch.full((1,), args.organism_index, dtype=torch.long, device=dev)

    @torch.no_grad()
    def validate():
        """Per-track Pearson r vs the mean-profile baseline, assembled ACROSS RANKS.

        Each rank holds only its positional slice of every validation window, so a locally
        computed correlation would describe a different subset on every rank and the two arms
        would not be comparable. Pearson is therefore built from sufficient statistics that are
        all-reduced: this returns exactly the value a single unsharded process would compute.

        TWO PASSES, deliberately. The one-pass form (n*Sxy - Sx*Sy) suffers catastrophic
        cancellation when n is large and values are O(1) -- which is exactly this regime. Global
        means are reduced first, then centred products, which is stable.

        The baseline is the mean profile over validation windows: a model that merely learns the
        average landscape scores well on within-window correlation, so beating this is the
        minimum bar for having learned anything sequence-specific.
        """
        model.eval(); head.eval()
        Ys, Ps = [], []
        for i in range(min(args.val_batches, len(ds_va))):
            xv_np, yv_np = ds_va[i]
            xv = torch.from_numpy(xv_np).unsqueeze(0).to(dev)
            yv = torch.from_numpy(yv_np).unsqueeze(0).to(dev) * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if seqpar:
                    e1v, _e128v, _p, _n = sp.forward(model, xv, org, resolutions=(1, 128),
                                                     original_length=args.seq_len)
                else:
                    ov = model(xv, org, embeddings_only=True, channels_last=False,
                               resolutions=(1, 128))
                    e1v = ov["embeddings_1bp"]
                pv = head({1: e1v}, org, return_scaled=True, channels_last=True)[1]
            # compare on the UNIQUE tracks only: replicated channels are copies and would just
            # repeat the same correlation T/n_uniq times, inflating nothing but the count
            Ps.append(torch.exp(pv[..., :n_uniq].float()).squeeze(0).cpu())
            Ys.append(yv.squeeze(0).cpu())
        model.train(); head.train()
        if not Ys:
            return {}
        Y = torch.cat(Ys, 0).double()                 # (N*L_shard, U)
        P = torch.cat(Ps, 0).double()
        M = torch.stack(Ys, 0).double().mean(0).repeat(len(Ys), 1)   # mean profile baseline

        def reduce_(t):
            if seqpar:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t

        n = reduce_(torch.tensor([float(Y.shape[0])], device=dev)).item()
        sums = reduce_(torch.stack([Y.sum(0), P.sum(0), M.sum(0)]).to(dev))
        my, mp, mm = sums[0] / n, sums[1] / n, sums[2] / n
        yc = Y - my.cpu(); pc = P - mp.cpu(); mc = M - mm.cpu()
        acc = reduce_(torch.stack([(yc * yc).sum(0), (pc * pc).sum(0), (mc * mc).sum(0),
                                   (yc * pc).sum(0), (yc * mc).sum(0)]).to(dev))
        vyy, vpp, vmm, cyp, cym = acc
        r_model = cyp / (vyy.sqrt() * vpp.sqrt()).clamp_min(1e-12)
        r_mean = cym / (vyy.sqrt() * vmm.sqrt()).clamp_min(1e-12)
        return {"r_model_median": float(r_model.median()),
                "r_meanprofile_median": float(r_mean.median()),
                "beats_baseline": int((r_model > r_mean).sum()),
                "n_tracks": int(r_model.numel())}

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    hist, t0 = [], time.time()
    best_r = float("-inf")
    if lead:
        print("step | loss | s/step | peak_GiB", flush=True)

    from collections import defaultdict
    ph = defaultdict(float)

    def mark(t0, key):
        """Attribute elapsed time to a phase. Synchronises, so only used under --profile."""
        if not args.profile:
            return t0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ph[key] += t1 - t0
        return t1

    for step in range(1, args.steps + 1):
        tP = time.perf_counter()
        # seq-par: every rank takes the SAME window (different positions within it).
        # DDP: every rank takes a DIFFERENT window. That is the whole distinction.
        if loader is not None:
            xb, yb = next(it)                       # already a tensor, prefetched and pinned
        else:
            wi = ((step - 1) * world + rank) if args.ddp else (step - 1)
            xb, yb = ds[int(order[wi % len(order)])]
            xb, yb = torch.from_numpy(xb), torch.from_numpy(yb)
        tP = mark(tP, "data")
        x = xb.unsqueeze(0).to(dev, non_blocking=True)
        y1 = yb.unsqueeze(0).to(dev, non_blocking=True)
        y1 = y1 * scale
        y128 = y1.reshape(1, y1.shape[1] // 128, 128, n_uniq).mean(2)
        tP = mark(tP, "h2d")

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if seqpar:
                e1, e128, _pair, _need = sp.forward(model, x, org, resolutions=(1, 128),
                                                    original_length=args.seq_len)
            else:
                out = model(x, org, embeddings_only=True, channels_last=False,
                            resolutions=(1, 128))
                e1, e128 = out["embeddings_1bp"], out["embeddings_128bp"]
            tP = mark(tP, "trunk")
            pred = head({1: e1, 128: e128}, org, return_scaled=True,
                        channels_last=not args.head_ncl)
            tP = mark(tP, "head")
            if args.head_ncl:
                # head returned (B, T, S); the loss expects (B, S, T)
                pred = {k: v.transpose(1, 2) for k, v in pred.items()}
            loss = chunked_poisson(pred[1], y1, idx) + chunked_poisson(pred[128], y128, idx)
            tP = mark(tP, "loss")
        loss.backward()
        tP = mark(tP, "backward")
        if seqpar or args.ddp:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        tP = mark(tP, "allreduce")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        tP = mark(tP, "optimizer")

        if seqpar or args.ddp:
            lt = torch.tensor([float(loss)], device=dev)
            dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            step_loss = float(lt)
        else:
            step_loss = float(loss)

        vm = validate() if (args.val_every and step % args.val_every == 0) else None
        if vm and lead and args.save_final:
            # Save at EVERY validation, not only at the end. A multi-hour run that dies at
            # step 7,000 would otherwise leave a full history.json and no model. Temp-then-
            # rename so a killed job cannot leave a truncated .pt that looks loadable.
            for tag, keep in (("last", True),
                              ("best", vm["r_model_median"] >= best_r)):
                if not keep:
                    continue
                dst = args.save_final.with_name(args.save_final.stem + f"_{tag}.pt")
                tmp = dst.with_suffix(".pt.tmp")
                torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "step": step, "metrics": vm, "channels": T,
                            "seed": args.seed, "world": world}, tmp)
                tmp.replace(dst)
            if vm["r_model_median"] >= best_r:
                best_r = vm["r_model_median"]
                print(f"        new best r_model={best_r:.4f} at step {step}", flush=True)
        if lead:
            rec = {"step": step, "loss": step_loss, "s_per_step": (time.time() - t0) / step}
            if vm:
                rec.update(vm)
                print(f"  VAL step {step:>5}  r_model={vm['r_model_median']:.4f}  "
                      f"r_meanprofile={vm['r_meanprofile_median']:.4f}  "
                      f"beats={vm['beats_baseline']}/{vm['n_tracks']}", flush=True)
            hist.append(rec)
            if step <= 3 or step % args.log_every == 0:
                print(f"{step:>5} | {step_loss:.6f} | {(time.time()-t0)/step:.2f} | "
                      f"{torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)
                tag = "ddp" if args.ddp else ("seqpar" if seqpar else "single")
                (args.out / f"loss_{tag}_w{world}_T{T}_L{args.seq_len}.json").write_text(
                    json.dumps({"seqpar": seqpar, "world": world, "seq_len": args.seq_len,
                                "channels": T, "seed": args.seed,
                                "peak_gib": round(torch.cuda.max_memory_allocated()/1024**3, 2),
                                "history": hist}, indent=1) + "\n")

    if lead and args.save_final:
        args.save_final.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "steps": args.steps, "seqpar": seqpar, "world": world,
                    "seed": args.seed, "channels": T}, args.save_final)
        print(f"  saved final weights -> {args.save_final}", flush=True)
    if lead:
        wins = world if args.ddp else 1
        settled = hist[-1]["s_per_step"] if hist else 0.0
        mode = "ddp" if args.ddp else ("seqpar" if seqpar else "single")
        print(f"FINAL_PEAK_GPU_GIB={torch.cuda.max_memory_allocated()/1024**3:.2f} "
              f"channels={T} seq_len={args.seq_len} world={world} mode={mode}", flush=True)
        # per-WINDOW is the only fair throughput comparison: DDP finishes `world` windows per
        # step while sequence parallelism finishes one, so s/step alone favours seq-par.
        print(f"THROUGHPUT mode={mode} world={world} s_per_step={settled:.3f} "
              f"windows_per_step={wins} S_PER_WINDOW={settled/max(wins,1):.3f}", flush=True)
        if args.profile and ph:
            tot = sum(ph.values())
            print(f"\nPHASE BREAKDOWN over {args.steps} steps (total {tot:.1f}s, "
                  f"{tot/args.steps:.2f}s/step):", flush=True)
            for k, v in sorted(ph.items(), key=lambda kv: -kv[1]):
                print(f"  PHASE {k:<10} {v/args.steps:7.3f} s/step  {100*v/tot:5.1f}%", flush=True)
        print("SEQPAR_TRAIN_DONE", flush=True)
    if seqpar or args.ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
