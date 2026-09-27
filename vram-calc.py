#!/usr/bin/env python3
"""vram-calc.py — VRAM/RAM sizing for llama.cpp LLM inference.

Planning tool: estimates whether a model + context fits on your GPUs.
Two layouts:
  full-gpu    all weights on GPU (dense models, or enough cards)
  moe-hybrid  --n-cpu-moe style: shared/attention weights + KV on GPU,
              expert FFN weights in system RAM (MoE models only)

All figures are approximations for planning — verify with llama-bench.

Examples:
  python3 vram-calc.py --preset llama-3.1-8b --quant Q4_K_M --context 8192 --gpus 1 --vram 24
  python3 vram-calc.py --preset qwen3-235b-a22b --quant IQ4_XS --context 245760 \
      --kv-dtype q8_0 --gpus 2 --vram 24
"""
from __future__ import annotations

import argparse

GIB = 2.0 ** 30

# Effective bits per weight including scale overhead (llama.cpp quant formats).
BPW = {
    "F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.4,
    "Q4_K_M": 4.85, "Q4_K_S": 4.6, "IQ4_XS": 4.3, "Q3_K_M": 4.0,
    "IQ3_XS": 3.3, "Q2_K": 3.3, "IQ2_XS": 2.3,
}

# Bytes per KV-cache element by --kv-dtype.
KV_BPE = {"f16": 2.0, "q8_0": 1.06, "q4_0": 0.56}

# Rough public specs. Fields:
#   total_b, active_b   parameter counts (billions)
#   layers, kv_dim      layer count, kv_heads * head_dim
#   attn                2.0 = GQA (separate K and V), 1.0 = MLA (compressed cache)
#   swa                 fraction of effective cache when sliding windows dominate
PRESETS = {
    "llama-3.1-8b":    dict(total_b=8.0,  active_b=8.0,  layers=32, kv_dim=1024, attn=2.0, swa=1.0),
    "llama-3.3-70b":   dict(total_b=70.6, active_b=70.6, layers=80, kv_dim=1024, attn=2.0, swa=1.0),
    "qwen3-30b-a3b":   dict(total_b=30.5, active_b=3.3,  layers=48, kv_dim=512,  attn=2.0, swa=1.0),
    "qwen3-235b-a22b": dict(total_b=235,  active_b=22,   layers=94, kv_dim=512,  attn=2.0, swa=1.0),
    "glm-4.5-air":     dict(total_b=106,  active_b=12,   layers=46, kv_dim=512,  attn=2.0, swa=1.0),
    "deepseek-v3":     dict(total_b=671,  active_b=37,   layers=61, kv_dim=576,  attn=1.0, swa=1.0),
    "gpt-oss-120b":    dict(total_b=117,  active_b=5.1,  layers=36, kv_dim=512,  attn=2.0, swa=0.1),
}

CUDA_OVERHEAD_GIB = 0.6  # CUDA context + cuBLAS workspaces, rough


def weights_gib(total_b: float, quant: str) -> float:
    return total_b * 1e9 * BPW[quant] / 8.0 / GIB


def kv_bytes_per_token(layers: int, kv_dim: int, attn: float, kv_dtype: str, swa: float) -> float:
    return layers * kv_dim * attn * KV_BPE[kv_dtype] * swa


def compute_gib(active_b: float) -> float:
    """Activation buffers scale with active params (rough)."""
    return 0.4 + active_b * 0.06


def fmt(x: float) -> str:
    return f"{x:,.1f}"


def main() -> int:
    p = argparse.ArgumentParser(
        description="VRAM sizing for llama.cpp inference (planning estimates).",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--params", type=float, help="total params (billions), if no preset")
    p.add_argument("--active", type=float, help="active params (billions); default = total (dense)")
    p.add_argument("--layers", type=int)
    p.add_argument("--kv-dim", type=int, help="kv_heads * head_dim")
    p.add_argument("--mla", action="store_true", help="MLA attention (single compressed KV)")
    p.add_argument("--swa", type=float, default=1.0, help="sliding-window cache factor (e.g. 0.1)")
    p.add_argument("--quant", default="Q4_K_M", choices=sorted(BPW))
    p.add_argument("--kv-dtype", default="f16", choices=sorted(KV_BPE))
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--vram", type=float, default=24.0, help="VRAM per GPU (GiB)")
    p.add_argument("--ram", type=float, help="system RAM (GiB), for the MoE-hybrid check")
    p.add_argument("--mode", choices=["auto", "full", "moe"], default="auto")
    a = p.parse_args()

    if a.preset:
        s = PRESETS[a.preset]
        total_b = float(s["total_b"])
        active_b = float(s["active_b"])
        layers, kv_dim = int(s["layers"]), int(s["kv_dim"])
        attn, swa = float(s["attn"]), float(s["swa"])
        name = a.preset
    else:
        if not (a.params and a.layers and a.kv_dim):
            p.error("need --preset, or all of --params/--layers/--kv-dim")
        total_b, layers, kv_dim = a.params, a.layers, a.kv_dim
        active_b = a.active if a.active is not None else a.params
        attn = 1.0 if a.mla else 2.0
        swa = a.swa
        name = "custom"

    kv_bpt = kv_bytes_per_token(layers, kv_dim, attn, a.kv_dtype, swa)
    kv_gib = kv_bpt * a.context / GIB
    w_gib = weights_gib(total_b, a.quant)
    c_gib = compute_gib(active_b)
    pool_gib = w_gib + kv_gib + c_gib + CUDA_OVERHEAD_GIB
    usable = a.gpus * a.vram * 0.96  # per-driver usable fraction

    print(f"model: {name}  ({total_b}B total, {active_b}B active)")
    print(f"quant: {a.quant} ({BPW[a.quant]} bpw)   kv: {a.kv_dtype}   ctx: {a.context:,}")
    print(f"gpus: {a.gpus} x {a.vram} GiB -> {usable:,.1f} GiB usable")
    print()

    is_moe = active_b < total_b * 0.9

    # --- full-GPU layout ---
    print("== full-GPU layout ==")
    print(f"  weights: {fmt(w_gib)}   kv({a.context:,} tok): {fmt(kv_gib)}   "
          f"compute: {fmt(c_gib)}   overhead: {CUDA_OVERHEAD_GIB}")
    print(f"  total:   {fmt(pool_gib)} GiB across {a.gpus} GPU(s) "
          f"({fmt(pool_gib / a.gpus)} per GPU)")
    if pool_gib <= usable:
        print(f"  FITS with {fmt(usable - pool_gib)} GiB spare; "
              f"max context ~{int((usable - w_gib - c_gib - CUDA_OVERHEAD_GIB) / (kv_bpt / GIB)):,} tokens")
    else:
        print(f"  DOES NOT FIT — short by {fmt(pool_gib - usable)} GiB")
    print()

    # --- MoE-hybrid layout (--n-cpu-moe) ---
    if is_moe:
        shared_b = max(active_b * 0.3, total_b * 0.02)  # heuristic, override in code if you know better
        shared_gib = weights_gib(shared_b, a.quant)
        expert_gib = weights_gib(total_b - shared_b, a.quant)
        vram_gib = shared_gib + kv_gib + c_gib + CUDA_OVERHEAD_GIB
        print("== MoE-hybrid layout (--n-cpu-moe) ==")
        print(f"  VRAM pool: shared weights {fmt(shared_gib)} + kv {fmt(kv_gib)} "
              f"+ compute {fmt(c_gib)} + overhead {CUDA_OVERHEAD_GIB} = {fmt(vram_gib)} GiB")
        print(f"  system RAM: expert weights ~{fmt(expert_gib)} GiB")
        if a.ram:
            ok = "ok" if expert_gib <= a.ram * 0.9 else "TOO LITTLE RAM"
            print(f"  ram check ({a.ram} GiB): {ok}")
        if vram_gib <= usable:
            max_ctx = int((usable - shared_gib - c_gib - CUDA_OVERHEAD_GIB) / (kv_bpt / GIB))
            print(f"  FITS; max context ~{max_ctx:,} tokens on this VRAM")
        else:
            print(f"  DOES NOT FIT — VRAM pool short by {fmt(vram_gib - usable)} GiB")
    else:
        print("== MoE-hybrid layout ==")
        print("  dense model — hybrid layout does not apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())