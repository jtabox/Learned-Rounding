# TPEC-Quant FP8 (Auto)

One-file, no-knobs converter that turns a safetensors model into a Scaled FP8 checkpoint compatible with ComfyUI. It uses a fast SVD-based “Top-Principal Error Correction” (TPEC) learned rounding scheme and automatically chooses good parameters per layer, so you don’t have to tune anything.

- Works great for Flux/Chroma-style image models.
- Optional T5-XXL mode: removes unnecessary decoder heads and writes extra metadata as expected by UIs.

## What it does

- Converts 2D weight tensors to FP8 (e4m3fn) with a per-tensor scale (stored as `.scale_weight`).
- Automatically picks a per-layer top_k (principal components) via a quick SVD probe to capture ≥90% of the layer’s energy, capped at 3 for speed.
- Optimizes rounding in the top principal subspace (fast TPEC) for 500 iterations per tensor.
- Applies bias correction using 3072 random calibration samples to reduce bias drift.
- Writes a `scaled_fp8` marker so ComfyUI uses the scaled-FP8 code path.
- T5-XXL mode: removes `decoder/*` and `lm_head*`, skips norms/embeddings/biases, and writes `.scale_input` keys (as expected by UIs).

## Quick start

- Flux/Chroma-style model:
  - `python autoconvert_fp8_gpt5pro.py --input /path/to/model.safetensors`
- T5-XXL:
  - `python autoconvert_fp8_gpt5pro.py --input /path/to/t5xxl.safetensors --t5xxl`
- Optional custom output path:
  - `python autoconvert_fp8_gpt5pro.py --input /path/to/model.safetensors --output /path/to/out.safetensors`

The script auto-selects CUDA if available. Defaults are set for good quality and speed; there’s nothing else to tweak.

## Output format (ComfyUI-compatible)

For each weight tensor `X.weight`:

- Stores FP8 weights (dtype: `float8_e4m3fn`).
- Stores `X.scale_weight` (shape `[1]`, dtype `float32`) for dequant.
- Keeps original non-weight tensors in their original dtype.
- Writes a marker key:
  - `scaled_fp8 = torch.empty((2), dtype=float8_e4m3fn)` (Flux/Chroma)
  - `scaled_fp8 = torch.empty((0), dtype=float8_e4m3fn)` (T5-XXL)

T5-XXL mode also:

- Removes `decoder/*` and `lm_head*` tensors from the output (UIs don’t need them).
- Adds `X.scale_input` (equal to `scale_weight`) when present in the model.

## How it works (in one paragraph)

Weights are scaled to fit FP8 range, then a quick SVD probe chooses a small top_k (≤3) capturing ≥90% of the layer’s Frobenius energy. The converter optimizes rounding in that subspace (TPEC) with a simple adaptive update and clamps to the FP8 representable range each step. After optimization, weights are stored in FP8 and a dequant scale (1/scale) is saved. Bias correction uses random calibration inputs to subtract the average propagated weight error from each bias.

## Defaults

- FP8 dtype: `float8_e4m3fn` (scaled per tensor)
- Energy threshold for SVD: 0.90 (top_k capped at 3)
- Iterations per tensor: 500
- Calibration samples: 3072 (bias correction)
- Device: auto (CUDA if available)

## Requirements

- Python 3.9+
- PyTorch with FP8 support (e.g., 2.1+ recommended)
- safetensors (recent version)

Install:

```bash
pip install torch safetensors
```

## Notes and limitations

- Only 2D `.weight` tensors are quantized; others are preserved.
- Calibration inputs are random; this is an offline quantization heuristic (not training).
- If you see “float8 not supported” errors, update PyTorch and safetensors.
- If you hit CUDA OOM on very large models, run on CPU (slower) or close other GPU workloads.
