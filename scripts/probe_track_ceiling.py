#!/usr/bin/env python3
"""How many output tracks fit at a given sequence length, with the sequence sharded across GPUs?

A sizing tool. Run it before committing to a track count, because the answer depends on world
size in a way DDP's does not: DDP replicates the model so its ceiling is fixed, while sequence
parallelism shards activations so the ceiling scales roughly with the number of GPUs.

Targets are allocated from the ACTUAL returned embedding shapes rather than assumed ones, because
the sequence-parallel path may hand back local shards (with overlap context) rather than the
gathered full-length tensor. Assuming would silently mis-size the loss.

Launch:
    torchrun --nproc_per_node=4 -m scripts.probe_track_ceiling --tracks 4000 8000 12000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn


# see the assert in one(): overlap_highres must equal overlap_lowres * 128, because the
# encoder downsamples the 1bp shard by 128 before gather_full trims in the 128bp domain.
OVERLAP_LOW = 32
OVERLAP_HIGH = OVERLAP_LOW * 128   # 4096


def make(num_tracks: int, grad_ckpt: bool, device: str):
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.heads import GenomeTracksHead

    m = AlphaGenome(num_organisms=2, gradient_checkpointing=grad_ckpt)
    m.heads = nn.ModuleDict()
    m.contact_maps_head = None
    m.splice_sites_classification_head = None
    m.splice_sites_usage_head = None
    m.splice_sites_junction_head = None
    for p in m.embedder_pair.parameters():   # feeds the removed contact-maps head
        p.requires_grad_(False)

    # The sequence-parallel forward calls model.embedder_pair(...) UNCONDITIONALLY and
    # returns a pairwise (B, S, S, D) tensor -- quadratic in the number of 128bp positions.
    # We predict 1-D tracks and discard it, so at 1 Mb that allocation is both large and
    # pure waste. Stub it out: this is what production 1-D-track training should also do.
    class _NoPair(nn.Module):
        def forward(self, pair_activations, organism_index):  # noqa: ARG002
            return torch.zeros(1, device=pair_activations.device)

    m.embedder_pair = _NoPair()
    h = GenomeTracksHead(in_channels=None, num_tracks=num_tracks,
                         resolutions=(1, 128), num_organisms=2)
    return m.to(device), h.to(device)


def one(L: int, T: int, ckpt: bool, steps: int, device: str, rank: int) -> dict:
    from alphagenome_pytorch.sequence_parallel import create_sequence_parallel_strategy

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"seq_len": L, "num_tracks": T, "grad_ckpt": ckpt,
           "world_size": dist.get_world_size()}
    hold: dict = {}
    try:
        m, h = make(T, ckpt, device)
        hold.update(m=m, h=h)
        # 🚨 The package DEFAULTS (overlap_highres=1024, overlap_lowres=32) are internally
        # inconsistent and make gather_full assert. The 1bp input is sharded with
        # overlap_highres, then the encoder downsamples by 128, so the effective overlap in
        # the 128bp domain is overlap_highres/128 = 8 -- but gather_full trims by
        # overlap_lowres = 32. The 4 shards then lose 144 positions in total, exactly the
        # observed "Expected length 8192, got 8048".
        #
        # The invariant is:  overlap_highres == overlap_lowres * 128
        assert OVERLAP_HIGH == OVERLAP_LOW * 128, "sequence-parallel overlaps must agree"
        sp = create_sequence_parallel_strategy(overlap_highres=OVERLAP_HIGH,
                                               overlap_lowres=OVERLAP_LOW)
        opt = torch.optim.Adam(
            [p for p in list(m.parameters()) + list(h.parameters()) if p.requires_grad],
            lr=1e-4)
        hold["opt"] = opt

        x = torch.randn(1, L, 4, device=device)
        org = torch.zeros(1, dtype=torch.long, device=device)
        hold.update(x=x, org=org)

        ts = []
        for i in range(steps + 1):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # NB forward returns FOUR values, not the two its type hint declares:
                # (embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp), and the
                # embeddings are LOCAL shards (already trimmed to this rank's region).
                e1, e128, _pair, _need = sp.forward(
                    m, x, org, resolutions=(1, 128), original_length=L)
                emb = {1: e1, 128: e128}
                if i == 0 and rank == 0:
                    rec["emb_shapes"] = {str(k): tuple(v.shape) for k, v in emb.items()}
                pred = h(emb, org, return_scaled=True, channels_last=True)
                # size the loss from what we actually got back
                loss = sum(p.float().mean() for p in pred.values())
            loss.backward(); opt.step(); torch.cuda.synchronize()
            if i:
                ts.append(time.time() - t0)
        ts.sort()
        rec.update(ok=True, peak_GB=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                   step_s=round(ts[len(ts) // 2], 3))
    except torch.cuda.OutOfMemoryError:
        rec.update(ok=False, error="OOM")
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:180])
    finally:
        hold.clear(); gc.collect()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # a rank that OOMs while others survive must not deadlock the sweep
    flag = torch.tensor([0 if rec.get("ok") else 1], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() and rec.get("ok"):
        rec.update(ok=False, error="OOM on another rank")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON summary; results also print to stdout")
    ap.add_argument("--steps", type=int, default=2,
                    help="forward+backward steps per candidate. 2 is enough: this measures peak "
                         "memory, not convergence.")
    ap.add_argument("--tracks", type=int, nargs="+", default=[128, 1024, 2176, 4992],
                    help="output-channel counts to try, ascending. The probe reports which fit "
                         "and which OOM, so bracket your intended count.")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    if rank == 0:
        tot = torch.cuda.get_device_properties(local).total_memory / 1024**3
        print(f"world_size={world}  GPU={torch.cuda.get_device_name(local)}  {tot:.1f} GB\n",
              flush=True)

    results = []
    for L in (524_288, 1_048_576):
        for T in args.tracks:
            r = one(L, T, True, args.steps, dev, rank)
            if rank == 0:
                msg = (f"{r['peak_GB']:>6.2f} GB {r['step_s']:>6.3f}s"
                       if r["ok"] else f"  {r['error']}")
                print(f"  L={L:>9,} T={T:>5} ckpt=1 {msg}"
                      f"{'   ' + str(r.get('emb_shapes')) if r.get('emb_shapes') else ''}",
                      flush=True)
                results.append(r)

    if rank == 0 and args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"world_size": world, "results": results}, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
