# TPEC Tuning Guide

Tweaking a few defaults can tilt the trade-off toward faster conversions, higher quality, lower VRAM, or more deterministic behavior depending on the model family and your hardware.

## Key parameters and their effects

### `topk-max` (default 3)

- **What**: Caps how many principal components the optimizer uses.
- **Raise** to 4 for tough, high-rank layers (large FFNs, MoE, some attention blocks) to squeeze a bit more quality.
- **Lower** to 1 for speed-first runs.
- **Impact**: +k increases both the SVD probe cost and each optimization step roughly linearly in k. Expect +10–30% runtime per extra component on big layers.

### `energy-threshold` (default 0.90)

- **What**: Target fraction of Frobenius energy to capture in the SVD probe.
- **Raise** to 0.95 to recommend slightly larger k (better quality on spread spectra).
- **Drop** to 0.85 to bias toward smaller k (faster).
- **Impact**: Subtle but consistent; higher threshold nudges more layers to k=2–3.

### `auto-topk` mode (`per-layer` | `global` | `off`)

- **per-layer** (default): Best quality; adapts k per tensor but does an SVD probe per layer.
- **global**: Samples ~20 layers to pick one median k for all layers. Faster and more deterministic, small quality drop if your model has mixed spectra.
- **off**: Use k=1 unless you set --top-k manually; fastest, most predictable, biggest quality drop on “hard” layers.

### `num-iter` (default 500)

- **What**: Optimization iterations per tensor.
- 250–400 is often enough for conv/attention; 500–1000 helps stubborn FFNs.
- **Impact**: Linear runtime scaling; diminishing returns after ~500 on most layers.

### `calib-samples` (default 3072)

- **What**: Random inputs for bias correction stability.
- 1024 is fine for speed runs; 4096–8192 for text models (T5) or if you notice bias drift.
- **Impact**: Linear cost and memory with samples.

### `probe-k` (default 8)

- **What**: The rank used during SVD probing.
- Keep at 8 if topk-max ≤ 3. Increase to 12–16 only if you also raise topk-max and have very spread spectra.
- **Impact**: Affects probe time, not the optimization loop.

## When to tweak for speed vs quality

### Speed-first (quick validation or limited VRAM)

- **Flags**:
  - `--quality fast`
  - `--auto-topk off`
  - `--top-k 1`
  - Optionally `--calib-samples 1024`
- **Why**: Minimal SVD work, fewer iterations, smaller calibration batches.

### High-quality (release build)

- **Flags**:
  - `--quality high`
  - `--auto-topk per-layer`
  - `--topk-max 4`
  - `--energy-threshold 0.95`
  - Optionally `--calib-samples 8192`
- **Why**: More aggressive optimization and slightly larger subspaces improve “hard” layers.

### Deterministic and faster than per-layer

- **Flags**:
  - `--auto-topk global`
  - Optionally `--topk-max 2`
- **Why**: One k for the whole model from a quick sample. Good balance of speed and predictability.

### Model-specific suggestions

- **Flux/Chroma** (image diffusion, U-Nets)

  - Defaults usually solid: `--quality balanced`, `--auto-topk per-layer` is great.
  - If you see banding/color shifts: try `--topk-max 4` or `--quality high`.
  - For quick checks: `--auto-topk off` `--top-k 1` `--quality fast`.

- **T5-XXL** (text)

  - Many layers are close to low-rank; k=1 or 2 often suffices.
  - Speed-leaning:
    - `--auto-topk global` `--topk-max 2`
    - `--energy-threshold 0.85`
    - `--calib-samples 4096`
  - Quality-leaning:
    - `--auto-topk per-layer` `--topk-max 3`
    - `--energy-threshold 0.90-0.95`
    - `--calib-samples 4096-8192`

- **Low-rank/adapter-heavy** (LoRA-style)

  - Use `--auto-topk off` `--top-k 1` or `--auto-topk global` with `--topk-max 1`.
  - You can also drop `--num-iter` to 250 for speed.

- **Very large FFNs or MoE blocks** (spread spectra)
  - Use `--auto-topk per-layer` `--topk-max 4` `--energy-threshold 0.95`
  - Consider `--quality high` for tough models.

### VRAM considerations

- Calibration memory is sum over unique `in_features` of (`calib_samples` × `in_features` × 4 bytes).
- If you hit OOM:
  - Lower `--calib-samples` (e.g., 1024–2048).
  - Use `--auto-topk global` or `off` to reduce per-layer SVD probes.
  - As a last resort, run on CPU (much slower):
  ```bash
  # Linux
  CUDA_VISIBLE_DEVICES= python fp8_tpec_conv.py
  ```
  or
  ```bat
  rem Windows
  setx CUDA_VISIBLE_DEVICES ""
  python fp8_tpec_conv.py
  ```

### Example commands

- **Speed run** (most models):
  - `python fp8_tpec_conv.py --input model.safetensors --quality fast --auto-topk off --top-k 1`
- **High quality** (hard models):
  - `python fp8_tpec_conv.py --input model.safetensors --quality high --auto-topk per-layer --topk-max 4 --energy-threshold 0.95 --calib-samples 8192`
- **T5-XXL** speed-leaning:
  - `python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl --auto-topk global --topk-max 2 --energy-threshold 0.85 --calib-samples 4096`
- **Deterministic global setting**:
  - `python fp8_tpec_conv.py --input model.safetensors --auto-topk global`

## Ready-made profiles

Here are turnkey profiles with recommended parameters. Each includes the exact command, why it’s set up that way, and quick notes on speed/VRAM.

### Flux/Chroma — Highest quality

- **Command**:
  - `python fp8_tpec_conv.py --input model.safetensors --quality high --auto-topk per-layer --topk-max 4 --energy-threshold 0.95 --calib-samples 8192 --keep-distillation`
- **Why**:
  - Per-layer top_k for best layer-wise fit; raised cap (4) and higher energy threshold (0.95) capture tougher spectra.
  - High preset (1000 iters, 8192 samples) squeezes more error down; keep-distillation avoids degrading guidance layers.
- **Notes**:
  - Highest VRAM and runtime of these profiles. If you hit OOM, reduce --calib-samples to 4096.

### Flux/Chroma — Fastest conversion

- **Command**:
  - `python fp8_tpec_conv.py --input model.safetensors --quality fast --auto-topk off --top-k 1 --calib-samples 1024`
- **Why**:
  - Disables SVD per-layer probes; uses k=1 everywhere.
  - Fast preset cuts iterations and calibration size.
- **Notes**:
  - Biggest speedup; small quality drop on some FFN/attention layers. If you want a tiny quality bump with minimal cost, try --auto-topk global instead of off.

### T5XXL — Highest quality

- **Command**:
  - `python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl --quality high --auto-topk per-layer --topk-max 3 --energy-threshold 0.90 --calib-samples 4096`
- **Why**:
  - T5 layers are often low-rank; cap 3 is enough. Per-layer auto preserves quality without overspending on rank.
  - High preset plus moderate calibration (4096) balances quality and VRAM for big text models.
- **Notes**:
  - If you want the absolute best and have headroom, bump `--energy-threshold 0.95` and `--calib-samples 8192`.

### T5XXL — Fastest conversion

- **Command** (recommended fast/deterministic):
  - `python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl --quality fast --auto-topk global --topk-max 2 --energy-threshold 0.85 --calib-samples 2048`
- **Alternate** (max speed):
  - `python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl --quality fast --auto-topk off --top-k 1 --calib-samples 2048`
- **Why**:
  - Global mode picks one k from a quick sample—faster than per-layer, more quality than fully off. Capping at 2 is usually enough for T5.
  - The alternate “off + k=1” is the absolute fastest with more quality trade-off.
- **Notes**:
  - T5 mode automatically removes decoder/lm_head and adds .scale_input; this also speeds up saving.

## Quick tips

- **If you see OOM**: lower `--calib-samples` (e.g., 2048), or switch `--auto-topk` to `global`/`off`.
- **If you see visible quality issues on Flux/Chroma with the fast profile**: try `--auto-topk global` or raise `--top-k` to 2.
- **For a small extra quality boost with modest overhead**: keep global mode and set `--topk-max 2` (Flux) or `2–3` (T5).
