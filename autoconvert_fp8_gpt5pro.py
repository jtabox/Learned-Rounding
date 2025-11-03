#!/usr/bin/env python3
"""
convert_fp8_tpec_auto.py

All-in-one, minimal-knobs FP8 converter using fast SVD/TPEC learned rounding.

- Auto-selects per-tensor top_k via quick SVD energy capture (90% default).
- Uses an efficient PCA (torch.pca_lowrank) for top singulars and for optimization basis.
- Applies bias correction on random calibration inputs (default 3072 samples).
- Optimized for Flux/Chroma models by default; supports T5-XXL via --t5xxl.
- Writes a Scaled FP8 safetensors file compatible with ComfyUI (adds "scaled_fp8" marker).

Usage:
  python convert_fp8_tpec_auto.py --input /path/to/model.safetensors
  python convert_fp8_tpec_auto.py --input /path/to/t5xxl.safetensors --t5xxl

Notes:
- Device is auto-selected (CUDA if available).
- Internal defaults: num_iter=500, top_k chosen automatically (1..3), calib_samples=3072.
- You can optionally pass --output to set the output path.
"""

import argparse
import os
import gc
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# -----------------------
# Configuration (minimal)
# -----------------------
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# T5-specific handling
AVOID_KEY_NAMES_T5 = ["norm", "bias", "embed_tokens", "shared"]
REMOVE_KEY_NAMES_T5 = ["decoder", "lm_head"]

# Auto top_k chooser
ENERGY_THRESHOLD = 0.90  # target energy capture for SVD (sigma^2 basis)
PROBE_K = 8  # probe up to this many singular values
TOPK_MAX = 3  # cap top_k for speed/robustness
NUM_ITER = 500  # optimization iterations
CALIB_SAMPLES = 3072  # random samples for bias correction


# -----------------------
# FP8 constants
# -----------------------
def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)


FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)


# -----------------------
# Converter (fast SVD/TPEC)
# -----------------------
class LearnedRoundingConverterTPEC:
    """
    Fast SVD/TPEC learned rounding on a single 2D weight matrix.
    - Works in a scaled domain (per-tensor scalar scaling).
    - Projects error onto top principal directions and optimizes there.
    - Final cast to FP8 with stored dequant scale.
    """

    def __init__(self, device: str = None, num_iter: int = NUM_ITER):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_iter = num_iter
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"Initialized TPEC converter on device: {self.device}")

    @torch.no_grad()
    def convert(
        self, W_orig: torch.Tensor, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a single weight tensor to Scaled FP8 with learned rounding.
        Returns:
          - W_f8 (FP8 tensor on CPU),
          - dequant_scale (1/scale) on CPU,
          - dequantized_weight (float32 on CPU).
        """
        assert W_orig.ndim == 2, "convert expects a 2D weight matrix"
        device = self.device

        W_f32 = W_orig.to(device, dtype=COMPUTE_DTYPE)
        w_max = W_f32.abs().max()
        if w_max < 1e-12:
            # all-zero or negligible tensor
            deq = torch.tensor([1.0], device=device, dtype=COMPUTE_DTYPE)
            W_f8 = torch.zeros_like(W_f32, dtype=TARGET_FP8_DTYPE)
            return (
                W_f8.cpu(),
                deq.cpu(),
                torch.zeros_like(W_f32, dtype=COMPUTE_DTYPE).cpu(),
            )

        # per-tensor scalar scaling to fit FP8 range
        scale = self.f8_max_val / w_max
        W_scaled = W_f32 * scale

        # initial (round-to-nearest) in scaled domain
        W_rounded = W_scaled.to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        W_q = W_rounded.clone()

        # get top_k principal components
        m, n = W_f32.shape
        k = int(max(1, min(top_k, TOPK_MAX, m, n)))
        try:
            U, _, V = torch.pca_lowrank(W_f32, q=k, center=False, niter=16)
            Vh = V.T
        except Exception:
            U, _, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            U = U[:, :k]
            Vh = Vh[:k, :]

        # simple adaptive step schedule
        best_loss = float("inf")
        best_tensor = W_q.clone()
        worse_counter = 0
        lr = 1.0
        curr_lr = lr

        pbar = tqdm(range(self.num_iter), desc="    TPEC optimize", leave=False)
        for i in pbar:
            # dequantized current
            E = (W_q / scale) - W_f32  # error in unscaled domain

            # project error onto principal subspace
            proj = U.T @ E @ Vh.T  # shape (k,k)
            loss = torch.sum(proj * proj)  # ||proj||_F^2

            if loss.item() < 1e-8:
                pbar.set_postfix({"loss": f"{loss.item():.2e}", "note": "early-stop"})
                break

            # track best
            if loss.item() >= best_loss:
                worse_counter += 1
                curr_lr = max(curr_lr / 2, 1e-8)
                if worse_counter >= 40:
                    pbar.set_postfix({"loss": f"{best_loss:.2e}", "note": "keep-best"})
                    W_q = best_tensor
                    break
            else:
                best_loss = loss.item()
                best_tensor = W_q.clone()
                worse_counter = 0
                curr_lr = min(curr_lr * 2, 8.0)

            # gradient in unscaled domain: dL/dE = 2 U proj V
            grad_E = 2.0 * (U @ proj @ Vh)
            # map to scaled domain: E = W_q/scale - W, so dL/dW_q = (1/scale) * dL/dE
            grad_Wq = grad_E / scale

            # update in scaled domain
            W_q = W_q - curr_lr * grad_Wq

            # keep within representable scaled FP8 range
            W_q.clamp_(FP8_MIN, FP8_MAX)

            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        final_scaled = best_tensor
        W_f8 = final_scaled.to(TARGET_FP8_DTYPE)

        dequant_scale = (1.0 / scale).reshape(1)
        W_dequant = W_f8.to(COMPUTE_DTYPE) * dequant_scale

        # cleanup device memory
        del W_f32, W_scaled, W_rounded, W_q, U, Vh, final_scaled
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return W_f8.cpu(), dequant_scale.cpu(), W_dequant.cpu()


# -----------------------
# Utilities
# -----------------------
@torch.no_grad()
def auto_recommend_top_k(
    W_cpu: torch.Tensor,
    device: str,
    probe_k: int = PROBE_K,
    energy_threshold: float = ENERGY_THRESHOLD,
    topk_max: int = TOPK_MAX,
) -> Tuple[int, float]:
    """
    Quick SVD probe to recommend top_k for this tensor.
    Returns (k_recommended, top1_energy_frac).
    Energy computed with sigma^2 (Frobenius energy).
    """
    m, n = W_cpu.shape
    k_probe = int(max(1, min(probe_k, m, n)))
    if W_cpu.numel() == 0 or k_probe <= 0:
        return 1, 0.0

    W = W_cpu.to(device, dtype=COMPUTE_DTYPE)
    total_energy = torch.linalg.norm(W, ord="fro").pow(2).item()
    if total_energy <= 0:
        return 1, 0.0

    try:
        U, S, V = torch.pca_lowrank(W, q=k_probe, center=False, niter=8)
        S = S
    except Exception:
        S = torch.linalg.svdvals(W)[:k_probe]

    sigma_sq = S * S
    cum = torch.cumsum(sigma_sq, dim=0).cpu().numpy().tolist()
    thresh = energy_threshold * total_energy

    # find smallest k meeting threshold
    rec = k_probe
    for i, v in enumerate(cum):
        if v >= thresh:
            rec = i + 1
            break

    rec = int(max(1, min(rec, topk_max, m, n)))

    top1_energy_frac = (sigma_sq[0].item() / total_energy) if S.numel() > 0 else 0.0
    return rec, float(top1_energy_frac)


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    d = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            d[k] = f.get_tensor(k).cpu()
    return d


# -----------------------
# Main workflow
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="All-in-one FP8 Scaled TPEC converter (auto top_k, bias correction)."
    )
    parser.add_argument(
        "--input", required=True, help="Input safetensors model (float weights)."
    )
    parser.add_argument(
        "--output", help="Output safetensors path (if omitted, auto-named)."
    )
    parser.add_argument(
        "--t5xxl",
        action="store_true",
        help="Enable T5-XXL handling (remove decoder/lm_head, skip some weights).",
    )
    args = parser.parse_args()

    # Check input
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    # Dtype support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except Exception:
        print("Error: torch.float8_e4m3fn not supported by this PyTorch/hardware.")
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Output path
    fp8_type_str = TARGET_FP8_DTYPE.__str__().split(".")[-1]
    if not args.output:
        base = os.path.splitext(args.input)[0]
        suffix = "_t5xxl" if args.t5xxl else ""
        args.output = f"{base}_{fp8_type_str}_scaled_tpec_auto{suffix}.safetensors"
    if os.path.abspath(args.output) == os.path.abspath(args.input):
        print("Error: output path must differ from input path.")
        return

    print(f"Loading model: {args.input}")
    tensors = load_safetensors(args.input)
    print(f"Loaded {len(tensors)} tensors.")

    # Auto-detect T5 if not specified (heuristic: presence of 'decoder' or 'lm_head')
    auto_t5 = any((("decoder" in k) or ("lm_head" in k)) for k in tensors.keys())
    t5xxl = args.t5xxl or auto_t5
    if t5xxl and not args.t5xxl:
        print("Note: detected T5-like model; enabling T5-XXL handling automatically.")
    print(f"T5-XXL mode: {t5xxl}")

    # Pre-generate calibration inputs per in_features for bias correction
    calib_cache: Dict[int, torch.Tensor] = {}
    for k, t in tensors.items():
        if k.endswith(".weight") and t.ndim == 2 and t.numel() > 0:
            in_features = t.shape[1]
            if in_features not in calib_cache:
                calib_cache[in_features] = torch.randn(
                    CALIB_SAMPLES, in_features, dtype=COMPUTE_DTYPE, device=device
                )

    converter = LearnedRoundingConverterTPEC(device=device, num_iter=NUM_ITER)

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([k for k in tensors.keys() if k.endswith(".weight")])
    total = len(weight_keys)
    processed = 0
    skipped = 0
    removed = 0

    print(f"Found {total} weight tensors to consider.")
    for idx, key in enumerate(weight_keys, 1):
        W = tensors[key]
        if W.ndim != 2 or W.numel() == 0:
            # Store empty as FP8 and 1.0 scale for consistency
            base = key[: -len(".weight")]
            new_tensors[key] = W.to(TARGET_FP8_DTYPE)
            new_tensors[f"{base}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        # T5 removal/skip
        if t5xxl and any(rm in key for rm in REMOVE_KEY_NAMES_T5):
            print(f"({idx}/{total}) Removing T5 decoder tensor: {key}")
            removed += 1
            continue
        if t5xxl and any(av in key for av in AVOID_KEY_NAMES_T5):
            print(f"({idx}/{total}) Skipping (T5 exclude): {key}")
            new_tensors[key] = W  # keep original
            skipped += 1
            continue

        print(f"({idx}/{total}) Quantizing: {key}")
        # Recommend top_k
        rec_k, top1_frac = auto_recommend_top_k(W, device=device)
        print(f"    recommended top_k={rec_k} (top1_energy_frac={top1_frac:.3f})")

        # Convert
        W_f8, dequant_scale, W_deq = converter.convert(W, top_k=rec_k)

        # Store
        new_tensors[key] = W_f8
        base = key[: -len(".weight")]
        scale_key = f"{base}.scale_weight"
        new_tensors[scale_key] = dequant_scale.to(SCALE_DTYPE)

        # Bias correction
        bias_key = f"{base}.bias"
        if bias_key in tensors:
            b_orig = tensors[bias_key]
            # weight_error in compute device
            W_orig_dev = W.to(device, dtype=COMPUTE_DTYPE)
            W_deq_dev = W_deq.to(device, dtype=COMPUTE_DTYPE)
            X_dev = calib_cache[W.shape[1]]  # already on device
            b_dev = b_orig.to(device, dtype=COMPUTE_DTYPE)

            weight_error = W_orig_dev - W_deq_dev
            output_error = X_dev @ weight_error.T
            bias_correction = output_error.mean(dim=0)
            b_new = b_dev - bias_correction

            new_tensors[bias_key] = b_new.cpu().to(b_orig.dtype)

            # cleanup
            del (
                W_orig_dev,
                W_deq_dev,
                X_dev,
                b_dev,
                weight_error,
                output_error,
                bias_correction,
                b_new,
            )
            if device == "cuda":
                torch.cuda.empty_cache()
        if t5xxl:
            # For T5 paths, also add scale_input (Comfy convention in existing scripts)
            new_tensors[f"{base}.scale_input"] = (
                dequant_scale.detach().clone().to(SCALE_DTYPE)
            )

        processed += 1
        # try to keep memory tidy
        del W_f8, dequant_scale, W_deq
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Add non-quantized tensors (and skip removed ones if t5xxl)
    for k, t in tensors.items():
        if t5xxl and any(rm in k for rm in REMOVE_KEY_NAMES_T5):
            continue
        if k not in new_tensors:
            new_tensors[k] = t

    # Add FP8 marker
    new_tensors["scaled_fp8"] = (
        torch.empty((2), dtype=TARGET_FP8_DTYPE)
        if not t5xxl
        else torch.empty((0), dtype=TARGET_FP8_DTYPE)
    )

    # Save out
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving quantized model to: {args.output}")
    save_file(new_tensors, args.output)
    print("Done.\nSummary:")
    print(f"  processed weights : {processed}")
    print(f"  skipped (t5 excl) : {skipped}")
    print(f"  removed (t5 dec)  : {removed}")
    print(f"  total output tensors: {len(new_tensors)}")


if __name__ == "__main__":
    main()
