#!/usr/bin/env python3
"""
Benchmark fused MoE kernel configs for Qwen3.5-122B-A10B-AWQ on AMD gfx1151.

Uses vLLM's fused_experts_impl() to benchmark MoE performance with
different BLOCK_SIZE_M / GROUP_SIZE_M configs.

Model MoE params: E=256, N=1024, K=3072, top_k=8, AWQ int4 group_size=128.

Usage (inside container):
    python /tuning/benchmark_moe_configs.py --output-dir /tuning/moe-configs

Expected runtime: 30-90 minutes (Triton compilation dominates first runs).
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch


def set_moe_config(config):
    """Directly set the global MoE config override."""
    import vllm.model_executor.layers.fused_moe as fused_moe_module
    fused_moe_module._config = config


def benchmark_one(M, E, N, K, top_k, group_size, config_override,
                  warmup=2, repeat=5):
    """Benchmark fused_experts_impl with a specific config override."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    device = "cuda"

    # Activations [M, K] in float16
    A = torch.randn(M, K, device=device, dtype=torch.float16)

    # For int4_w4a16, assertion is: hidden_size // 2 == w1.size(2)
    # w1: [E, 2*N, K//2] (int4 packs 2 per element)
    # w2: [E, N, K//2]
    w1 = torch.randint(0, 2**15, (E, 2 * N, K // 2),
                       device=device, dtype=torch.int16).to(torch.int32)
    w2 = torch.randint(0, 2**15, (E, N, K // 2),
                       device=device, dtype=torch.int16).to(torch.int32)

    # Scales: [E, K // group_size, dim]
    w1_scale = torch.ones(E, K // group_size, 2 * N, device=device, dtype=torch.float16)
    w2_scale = torch.ones(E, K // group_size, N, device=device, dtype=torch.float16)

    # Routing
    topk_weights = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).half()
    topk_ids = torch.stack([
        torch.randperm(E, device=device)[:top_k] for _ in range(M)
    ]).int()

    set_moe_config(config_override)

    try:
        for _ in range(warmup):
            fused_experts_impl(
                A, w1, w2, topk_weights, topk_ids,
                inplace=False,
                use_int4_w4a16=True,
                w1_scale=w1_scale, w2_scale=w2_scale,
                block_shape=[0, group_size],
            )
        torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fused_experts_impl(
                A, w1, w2, topk_weights, topk_ids,
                inplace=False,
                use_int4_w4a16=True,
                w1_scale=w1_scale, w2_scale=w2_scale,
                block_shape=[0, group_size],
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        set_moe_config(None)
        del A, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
        torch.cuda.empty_cache()
        return sorted(times)[len(times) // 2]

    except Exception as e:
        set_moe_config(None)
        del A, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tuning/moe-configs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    E, N, K, TOP_K, GS = 256, 1024, 3072, 8, 128

    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print(f"MoE: E={E}, N={N}, K={K}, top_k={TOP_K}, group_size={GS}", flush=True)
    print(f"Output: {args.output_dir}\n", flush=True)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    configs_to_test = [
        {"BLOCK_SIZE_M": bm, "GROUP_SIZE_M": gm, "SPLIT_K": 1}
        for bm in [16, 32, 64, 128]
        for gm in [1, 4, 8, 16, 32]
    ]

    total = len(configs_to_test) * len(batch_sizes)
    print(f"{len(configs_to_test)} configs x {len(batch_sizes)} batch sizes = {total} benchmarks", flush=True)
    print("First call triggers Triton compilation (may take 5-10 min)...\n", flush=True)

    # Warmup compilation
    t0 = time.time()
    r = benchmark_one(4, E, N, K, TOP_K, GS,
                      {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1},
                      warmup=1, repeat=1)
    elapsed = time.time() - t0
    if r is None:
        print(f"Warmup FAILED after {elapsed:.0f}s", flush=True)
        # Try default
        r = benchmark_one(4, E, N, K, TOP_K, GS, None, warmup=1, repeat=1)
        if r is None:
            print("Default also failed. Exiting.", flush=True)
            sys.exit(1)
        print(f"Default works: {r:.3f}ms", flush=True)
    else:
        print(f"Compilation done: {r:.3f}ms ({elapsed:.0f}s)\n", flush=True)

    best_configs = {}
    done = 0

    for M in batch_sizes:
        print(f"\n--- M={M} ---", flush=True)
        best_time = float("inf")
        best_cfg = None

        for cfg in configs_to_test:
            done += 1
            r = benchmark_one(M, E, N, K, TOP_K, GS, cfg, warmup=2, repeat=5)
            if r is None:
                continue
            if r < best_time:
                best_time = r
                best_cfg = cfg.copy()
                print(f"  [{done}/{total}] {r:.3f}ms  M={cfg['BLOCK_SIZE_M']} G={cfg['GROUP_SIZE_M']}", flush=True)

        if best_cfg:
            best_configs[M] = best_cfg
            print(f"  >> Best M={M}: {best_time:.3f}ms  {best_cfg}", flush=True)
        else:
            print(f"  >> No valid config for M={M}", flush=True)

    out = {str(m): cfg for m, cfg in best_configs.items()}

    for fname in [
        "E=256,N=1024,device_name=AMD-gfx1151,dtype=int4_w4a16.json",
        "E=256,N=1024,device_name=AMD-gfx1151.json",
    ]:
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"Wrote: {path}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
