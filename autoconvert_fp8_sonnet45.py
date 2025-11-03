#!/usr/bin/env python3
"""
auto_fp8_convert.py

Simple, automated FP8 conversion using learned SVD quantization (TPEC-Quant method).
Automatically inspects your model and determines optimal parameters.

Usage:
    python auto_fp8_convert.py --input model.safetensors
    python auto_fp8_convert.py --input model.safetensors --t5xxl
    python auto_fp8_convert.py --input model.safetensors --output custom_output.safetensors
"""

import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc

# Configuration
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# Exclusion lists
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]


class LearnedRoundingConverter:
    """TPEC-Quant (Top-Principal Error Correction Quantization) converter."""

    def __init__(self, num_iter=500, top_k=1):
        self.num_iter = num_iter
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max

    def convert(
        self, W_orig: torch.Tensor, X_calib: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a weight tensor to FP8 using learned rounding."""
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        # Calculate quantization scale
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return (
                quantized_tensor.cpu(),
                scale.reciprocal().cpu().reshape(1),
                torch.zeros_like(W_float32).cpu(),
            )

        scale = self.f8_max_val / w_max
        W_scaled = W_float32 * scale

        # Naive round-to-nearest as starting point
        W_rounded = W_scaled.to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)

        # SVD/PCA for top principal components
        k = min(self.top_k, min(W_float32.shape))
        U, _, Vh = torch.pca_lowrank(W_float32, q=k, center=False, niter=16)
        Vh = Vh.T
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]

        W_q_refined = W_rounded.clone()

        # Optimization loop
        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        lr = 1.0
        curr_lr = lr

        pbar = tqdm(range(self.num_iter), desc="    Optimizing", leave=False)
        for i in pbar:
            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error) ** 2

            if loss.abs() < 1e-8:
                break

            # Simple learning rate scheduling and early stopping
            if loss.abs() >= best_loss:
                worse_loss_counter += 1
                curr_lr = max(curr_lr / 2, 1e-8)
                if worse_loss_counter >= 40:
                    break
            else:
                best_loss = loss.abs().item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = curr_lr * 2

            grad = U_k @ projected_error @ Vh_k
            W_q_refined = W_q_refined - curr_lr * grad
            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        final_tensor = best_tensor if best_tensor is not None else W_q_refined

        # Final quantization
        with torch.no_grad():
            W_f8 = final_tensor.to(TARGET_FP8_DTYPE)

        dequant_scale = scale.reciprocal().reshape(1)

        # Cleanup
        del W_float32, W_scaled, W_rounded, W_q_refined, error, U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (
            W_f8.cpu(),
            dequant_scale.cpu(),
            (W_f8.to(COMPUTE_DTYPE) * dequant_scale).cpu(),
        )


def quick_inspect_model(path: str, probe_k: int = 6) -> int:
    """
    Quickly inspect model to determine optimal global top_k.
    Returns recommended top_k for the entire model.
    """
    print("🔍 Analyzing model structure...")
    recommendations = []

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")]

        for key in tqdm(
            keys[: min(20, len(keys))], desc="Sampling layers"
        ):  # Sample first 20 layers
            W = f.get_tensor(key).to(torch.float32).cpu()
            if W.ndim != 2 or W.numel() == 0:
                continue

            m, n = W.shape
            k = min(probe_k, min(m, n))

            try:
                # Fast PCA to get top singular values
                _, S, _ = torch.pca_lowrank(W, q=k, center=False, niter=8)

                # Calculate energy fractions
                sigma_sq = S**2
                total_energy = torch.linalg.norm(W, ord="fro") ** 2

                # Find k where we capture 90% of energy
                cum_energy = torch.cumsum(sigma_sq, dim=0)
                threshold = 0.90 * total_energy

                for i, e in enumerate(cum_energy):
                    if e >= threshold:
                        recommendations.append(i + 1)
                        break
                else:
                    recommendations.append(k)

            except Exception:
                recommendations.append(1)

    if not recommendations:
        return 1

    # Use median recommendation, capped at 4 for speed
    recommended = min(int(sorted(recommendations)[len(recommendations) // 2]), 4)
    print(f"✓ Recommended top_k: {recommended}")
    return recommended


def convert_to_fp8(
    input_file: str,
    output_file: str,
    t5xxl: bool = False,
    keep_distillation: bool = False,
    auto_top_k: bool = True,
    manual_top_k: int = None,
    num_iter: int = 500,
    calib_samples: int = 3072,
):
    """Main conversion function."""

    print(f"\n{'=' * 60}")
    print("FP8 Auto-Converter (TPEC-Quant Method)")
    print(f"{'=' * 60}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Mode:   {'T5XXL' if t5xxl else 'Flux/Chroma'}")
    print(f"{'=' * 60}\n")

    # Load model
    print("📂 Loading model...")
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(input_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).cpu()
    print(f"✓ Loaded {len(tensors)} tensors\n")

    # Determine optimal top_k
    if manual_top_k is not None:
        top_k = manual_top_k
        print(f"Using manual top_k: {top_k}\n")
    elif auto_top_k:
        top_k = quick_inspect_model(input_file)
        print()
    else:
        top_k = 1
        print(f"Using default top_k: {top_k}\n")

    # Initialize converter
    print(f"⚙️  Initializing converter (top_k={top_k}, num_iter={num_iter})...")
    converter = LearnedRoundingConverter(num_iter=num_iter, top_k=top_k)
    print(f"✓ Running on: {converter.device}\n")

    # Generate calibration data
    print("📊 Generating calibration data...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples, in_features, dtype=COMPUTE_DTYPE
                )
    print(
        f"✓ Generated calibration data for {len(calibration_data_cache)} layer sizes\n"
    )

    # Process weights
    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith(".weight")])
    total_weights = len(weight_keys)
    processed_count = 0
    skipped_count = 0

    print(f"🔄 Processing {total_weights} weight tensors...\n")

    for i, key in enumerate(weight_keys):
        # Check exclusions
        should_skip = False

        if t5xxl and any(avoid in key for avoid in T5XXL_REMOVE_KEY_NAMES):
            print(f"[{i + 1}/{total_weights}] Removing: {key}")
            skipped_count += 1
            continue

        if t5xxl and any(avoid in key for avoid in AVOID_KEY_NAMES):
            print(f"[{i + 1}/{total_weights}] Skipping: {key}")
            new_tensors[key] = tensors[key]
            skipped_count += 1
            should_skip = True

        if keep_distillation and any(avoid in key for avoid in DISTILL_LAYER_KEYNAMES):
            print(f"[{i + 1}/{total_weights}] Skipping: {key}")
            new_tensors[key] = tensors[key]
            base_name = key[: -len(".weight")]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor(
                [1.0], dtype=SCALE_DTYPE
            )
            skipped_count += 1
            should_skip = True

        if should_skip:
            continue

        print(f"[{i + 1}/{total_weights}] Processing: {key}")
        processed_count += 1

        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print("  ⚠ Skipping empty/non-2D tensor")
            new_tensors[key] = tensors[key].to(TARGET_FP8_DTYPE)
            base_name = key[: -len(".weight")]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor(
                [1.0], dtype=SCALE_DTYPE
            )
            continue

        in_features = original_tensor.shape[1]
        calibration_data = calibration_data_cache.get(in_features)

        if calibration_data is None:
            print("  ⚠ No calibration data, skipping")
            new_tensors[key] = original_tensor
            skipped_count += 1
            processed_count -= 1
            continue

        # Convert
        quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = (
            converter.convert(original_tensor, calibration_data)
        )

        # Store results
        new_tensors[key] = quantized_fp8_tensor
        base_name = key[: -len(".weight")]
        new_tensors[f"{base_name}.scale_weight"] = dequant_scale.to(SCALE_DTYPE)

        # Bias correction
        bias_key = f"{base_name}.bias"
        if bias_key in tensors:
            print("  ✓ Correcting bias")
            with torch.no_grad():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                W_orig_dev = original_tensor.to(device, dtype=COMPUTE_DTYPE)
                W_dequant_dev = dequantized_weight_tensor.to(
                    device, dtype=COMPUTE_DTYPE
                )
                X_calib_dev = calibration_data.to(device, dtype=COMPUTE_DTYPE)
                b_orig_dev = tensors[bias_key].to(device, dtype=COMPUTE_DTYPE)

                weight_error = W_orig_dev - W_dequant_dev
                output_error = X_calib_dev @ weight_error.T
                bias_correction = output_error.mean(dim=0)
                b_new = b_orig_dev - bias_correction

                new_tensors[bias_key] = b_new.cpu().to(tensors[bias_key].dtype)

                del (
                    W_orig_dev,
                    W_dequant_dev,
                    X_calib_dev,
                    b_orig_dev,
                    weight_error,
                    output_error,
                    bias_correction,
                    b_new,
                )
                if device == "cuda":
                    torch.cuda.empty_cache()

        if t5xxl:
            new_tensors[f"{base_name}.scale_input"] = (
                dequant_scale.detach().clone().to(SCALE_DTYPE)
            )

        print(f"  ✓ Scale: {dequant_scale.item():.6e}")

    # Add remaining tensors
    for key, tensor in tensors.items():
        if t5xxl and any(avoid in key for avoid in T5XXL_REMOVE_KEY_NAMES):
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor

    # Add FP8 marker
    new_tensors["scaled_fp8"] = (
        torch.empty((2), dtype=TARGET_FP8_DTYPE)
        if not t5xxl
        else torch.empty((0), dtype=TARGET_FP8_DTYPE)
    )

    # Save
    print(f"\n💾 Saving {len(new_tensors)} tensors...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(new_tensors, output_file)

    print(f"\n{'=' * 60}")
    print("✅ Conversion Complete!")
    print(f"{'=' * 60}")
    print(f"Original tensors:  {len(tensors)}")
    print(f"Processed weights: {processed_count}")
    print(f"Skipped weights:   {skipped_count}")
    print(f"Final tensors:     {len(new_tensors)}")
    print(f"Output saved to:   {output_file}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic FP8 conversion with learned SVD quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Flux model (auto-detect parameters)
  python auto_fp8_convert.py --input flux_model.safetensors

  # For T5XXL models
  python auto_fp8_convert.py --input t5xxl_model.safetensors --t5xxl

  # Manual top_k override
  python auto_fp8_convert.py --input model.safetensors --top-k 2

  # Quick/experimental conversion
  python auto_fp8_convert.py --input model.safetensors --quality fast

  # High quality conversion (slower)
  python auto_fp8_convert.py --input model.safetensors --quality high
""",
    )

    parser.add_argument("--input", required=True, help="Input safetensors file")
    parser.add_argument(
        "--output", help="Output file (auto-generated if not specified)"
    )
    parser.add_argument("--t5xxl", action="store_true", help="T5XXL model mode")
    parser.add_argument(
        "--keep-distillation",
        action="store_true",
        help="Keep distillation layers unquantized",
    )

    # Advanced options (most users won't need these)
    parser.add_argument(
        "--top-k", type=int, help="Manual top_k override (default: auto-detect)"
    )
    parser.add_argument(
        "--no-auto", action="store_true", help="Disable auto-detection (use top_k=1)"
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=500,
        help="Optimization iterations (default: 500)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=3072,
        help="Calibration samples (default: 3072)",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Preset quality level (default: balanced)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return

    # Check FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("❌ Error: PyTorch version does not support torch.float8_e4m3fn")
        return

    # Generate output filename
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        suffix = "_nodistill" if args.keep_distillation else ""
        args.output = (
            f"{base_name}_float8_e4m3fn_scaled_learned_svd{suffix}.safetensors"
        )

    # Prevent overwriting input
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("❌ Error: Output file cannot be the same as input file")
        return

    # Quality presets
    quality_presets = {
        "fast": {"num_iter": 250, "calib_samples": 1024},
        "balanced": {"num_iter": 500, "calib_samples": 3072},
        "high": {"num_iter": 1000, "calib_samples": 8192},
    }

    if args.quality in quality_presets:
        preset = quality_presets[args.quality]
        num_iter = args.num_iter if args.num_iter != 500 else preset["num_iter"]
        calib_samples = (
            args.calib_samples
            if args.calib_samples != 3072
            else preset["calib_samples"]
        )
    else:
        num_iter = args.num_iter
        calib_samples = args.calib_samples

    # Run conversion
    convert_to_fp8(
        input_file=args.input,
        output_file=args.output,
        t5xxl=args.t5xxl,
        keep_distillation=args.keep_distillation,
        auto_top_k=not args.no_auto,
        manual_top_k=args.top_k,
        num_iter=num_iter,
        calib_samples=calib_samples,
    )


if __name__ == "__main__":
    main()
