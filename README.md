# gpu-inference-stack

OpenAI-compatible LLM serving on consumer GPUs with [llama.cpp](https://github.com/ggml-org/llama.cpp) — service templates, sizing math, and benchmark tooling for running large models on 24GB cards.

## What's here

| File | What it does |
|---|---|
| `vram-calc.py` | Planning calculator: does model + context + KV dtype fit on N GPUs? Full-GPU and MoE-hybrid (`--n-cpu-moe`) layouts, per-model presets |
| `templates/llama-server.service` | Hardened systemd unit — loopback bind, dedicated user, `Restart=on-failure` |
| `templates/llama-server.scm` | Same as a Guix Home shepherd user service (respawn, login-started) |
| `bench.sh` | `llama-bench` sweep: models × context lengths × KV dtype → CSV |

## Sizing rules of thumb

- **Dense models:** weights + KV cache + compute buffers must fit in VRAM. 8B @ Q4 ≈ 5 GiB; 70B @ Q4 ≈ 42 GiB.
- **MoE models:** you don't need all weights in VRAM. `--n-cpu-moe` keeps attention + shared weights + KV on GPU and streams expert FFN weights from system RAM — a 235B-A22B @ 4-bit with a 200k+ context fits on 2× 24GB with 256GB RAM.
- **KV cache is the silent killer.** Long contexts dominate VRAM; quantizing it (`--kv-cache-dtype q8_0`) halves the f16 footprint at negligible quality cost. MLA models (DeepSeek family) compress it ~10×.
- Throughput comes from batch; interactive latency comes from a small fast model. Run both.

All figures are planning estimates — verify with `bench.sh` before betting hardware.

## Quick start

```bash
python3 vram-calc.py --preset qwen3-235b-a22b --quant IQ4_XS --context 245760 --kv-dtype q8_0 --gpus 2 --vram 24
python3 vram-calc.py --preset llama-3.1-8b --quant Q4_K_M --context 8192
```

## Philosophy

Self-host instead of renting the cloud. Loopback binds, env-var config, restart-on-crash services, and everything auditable in files you own.

## Status

Templates in active use on my own boxes. PRs/issues welcome.