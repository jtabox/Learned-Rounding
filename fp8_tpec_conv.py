"""
fp8_tpec_conv.py

Super FP8 Scaled Converter (TPEC-Quant learned rounding)
- Per-layer or global auto top_k via SVD energy probe (default per-layer).
- Smart Auto profile picks quality, top_k strategy, caps, and calibration size
  based on model hardness and available VRAM (one-switch usage).
- No-grad, clamped updates inside the converter for stability and speed.
- Bias correction with on-device calibration inputs.
- Quality presets, richer CLI, T5 auto-detection, keep-distillation,
  ComfyUI-compatible output.
- OOM resilience: automatic retry with lighter settings if CUDA runs out of memory.

Usage:
  python fp8_tpec_conv.py --input model.safetensors
  python fp8_tpec_conv.py --input model.safetensors --quality high
  python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl
  python fp8_tpec_conv.py --input model.safetensors --auto-topk global
  python fp8_tpec_conv.py --input model.safetensors --top-k 2
"""

import argparse
import os
import gc
from typing import Dict, Tuple, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# T5-specific handling
AVOID_KEY_NAMES_T5 = ["norm", "bias", "embed_tokens", "shared"]
REMOVE_KEY_NAMES_T5 = ["decoder", "lm_head"]
# Optional: preserve these distillation layers (if --keep-distillation)
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]

# Auto top_k chooser defaults
DEFAULT_ENERGY_THRESHOLD = 0.90  # target energy capture for SVD (sigma^2 basis)
DEFAULT_PROBE_K = 8  # probe up to this many singular values
DEFAULT_TOPK_MAX = 3  # cap top_k for speed/robustness

# Optimization defaults
DEFAULT_NUM_ITER = 500  # optimization iterations
DEFAULT_CALIB_SAMPLES = 3072  # random samples for bias correction


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

    def __init__(self, device: Optional[str] = None, num_iter: int = DEFAULT_NUM_ITER):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_iter = num_iter
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"⚙️  Initialized TPEC converter on device: {self.device}")

    @torch.no_grad()
    def convert(
        self, W_orig: torch.Tensor, top_k: int, topk_cap: int = DEFAULT_TOPK_MAX
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
        k = int(max(1, min(top_k, topk_cap, m, n)))
        try:
            U, _, V = torch.pca_lowrank(W_f32, q=k, center=False, niter=16)
            Vh = V.T  # shape (k, n)
        except Exception:
            U_full, _, Vh_full = torch.linalg.svd(W_f32, full_matrices=False)
            U = U_full[:, :k]
            Vh = Vh_full[:k, :]

        # simple adaptive step schedule
        best_loss = float("inf")
        best_tensor = W_q.clone()
        worse_counter = 0
        lr = 1.0
        curr_lr = lr

        pbar = tqdm(range(self.num_iter), desc="    TPEC optimize", leave=False)
        for _ in pbar:
            # dequantized current
            E = (W_q / scale) - W_f32  # error in unscaled domain

            # project error onto principal subspace
            proj = U.T @ E @ Vh.T  # shape (k,k)
            loss = torch.sum(proj * proj).item()  # ||proj||_F^2

            if loss < 1e-8:
                pbar.set_postfix({"loss": f"{loss:.2e}", "note": "early-stop"})
                break

            # track best
            if loss >= best_loss:
                worse_counter += 1
                curr_lr = max(curr_lr / 2, 1e-8)
                if worse_counter >= 40:
                    pbar.set_postfix({"loss": f"{best_loss:.2e}", "note": "keep-best"})
                    W_q = best_tensor
                    break
            else:
                best_loss = loss
                best_tensor = W_q.clone()
                worse_counter = 0
                curr_lr = min(curr_lr * 2, 8.0)

            # gradient in unscaled domain: dL/dE = 2 U proj V
            grad_E = 2.0 * (U @ proj @ Vh)
            # map to scaled domain: E = W_q/scale - W, so dL/dW_q = (1/scale) * dL/dE
            grad_Wq = grad_E / scale

            # update in scaled domain and clamp to representable FP8 range
            W_q.sub_(curr_lr * grad_Wq).clamp_(FP8_MIN, FP8_MAX)

            pbar.set_postfix({"loss": f"{loss:.2e}"})

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
    probe_k: int = DEFAULT_PROBE_K,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    topk_max: int = DEFAULT_TOPK_MAX,
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
        _, S, _ = torch.pca_lowrank(W, q=k_probe, center=False, niter=8)
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


@torch.no_grad()
def quick_inspect_model_global_topk(
    path: str,
    probe_k: int = DEFAULT_PROBE_K,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    cap: int = DEFAULT_TOPK_MAX,
) -> int:
    """
    Sample up to 20 2D weight tensors and recommend a GLOBAL top_k
    using median k capturing ≥ energy_threshold of Frobenius energy.
    """
    print("🔍 Sampling model to recommend a global top_k...")
    recommendations = []

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")]
        for key in tqdm(keys[: min(20, len(keys))], desc="Sampling layers"):
            W = f.get_tensor(key).to(torch.float32).cpu()
            if W.ndim != 2 or W.numel() == 0:
                continue

            m, n = W.shape
            k_probe = int(max(1, min(probe_k, m, n)))
            try:
                _, S, _ = torch.pca_lowrank(W, q=k_probe, center=False, niter=8)
            except Exception:
                S = torch.linalg.svdvals(W)[:k_probe]

            sigma_sq = S * S
            total_energy = torch.linalg.norm(W, ord="fro") ** 2
            cum_energy = torch.cumsum(sigma_sq, dim=0)
            threshold = energy_threshold * total_energy

            picked = k_probe
            for i, e in enumerate(cum_energy):
                if e >= threshold:
                    picked = i + 1
                    break
            recommendations.append(int(picked))

    if not recommendations:
        print("✓ Recommended global top_k: 1")
        return 1

    recommended = min(int(sorted(recommendations)[len(recommendations) // 2]), cap)
    print(f"✓ Recommended global top_k: {recommended}")
    return recommended


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    d = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            d[k] = f.get_tensor(k).cpu()
    return d


# -----------------------
# Smart Auto profiling
# -----------------------
def get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return 0.0


@torch.no_grad()
def sample_model_hardness(
    path: str,
    probe_k: int,
    energy_threshold: float = 0.90,
    max_layers: int = 20,
) -> dict:
    # Returns summary stats to decide auto profile
    hard_count = 0
    total = 0
    top1_fracs = []
    rec_k_list = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")]
        for key in keys[: min(max_layers, len(keys))]:
            W = f.get_tensor(key)
            if W.ndim != 2 or W.numel() == 0:
                continue
            rec_k, top1 = auto_recommend_top_k(
                W.to(torch.float32).cpu(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                probe_k=probe_k,
                energy_threshold=energy_threshold,
                topk_max=8,  # probe space; final cap chosen later
            )
            total += 1
            top1_fracs.append(top1)
            rec_k_list.append(rec_k)
            if rec_k >= 3 or top1 < 0.55:
                hard_count += 1
    if total == 0:
        return {"hard_ratio": 0.0, "median_top1": 1.0, "median_rec_k": 1}
    top1_fracs.sort()
    rec_k_list.sort()
    return {
        "hard_ratio": hard_count / total,
        "median_top1": top1_fracs[len(top1_fracs) // 2],
        "median_rec_k": rec_k_list[len(rec_k_list) // 2],
    }


def choose_auto_params(input_file: str, t5xxl_detected: bool) -> dict:
    free_gb = get_free_vram_gb()
    stats = sample_model_hardness(input_file, probe_k=8, energy_threshold=0.90)
    hard = stats["hard_ratio"] >= 0.35 or stats["median_rec_k"] >= 3

    # Base decisions
    if torch.cuda.is_available():
        if free_gb >= 12 and hard:
            quality = "high"
        elif free_gb >= 8:
            quality = "balanced"
        else:
            quality = "fast"
    else:
        quality = "fast"  # CPU fallback

    # Auto-topk mode
    if quality == "fast" or free_gb < 6 or not torch.cuda.is_available():
        auto_topk_mode = "global" if free_gb >= 4 else "off"
    else:
        auto_topk_mode = "per-layer"

    # Caps and thresholds
    if quality == "high" and hard and free_gb >= 12:
        topk_max = 4
        energy_threshold = 0.95
    elif quality == "fast":
        topk_max = 2
        energy_threshold = 0.85
    else:
        topk_max = 3
        energy_threshold = 0.90

    # Calibration sizes (Flux benefits from larger; T5 is fine moderate)
    if t5xxl_detected:
        calib = 4096 if (quality != "fast" and free_gb >= 10) else 2048
    else:
        calib = (
            8192
            if (quality == "high" and free_gb >= 12)
            else (3072 if quality == "balanced" else 1024)
        )

    return dict(
        quality=quality,
        auto_topk_mode=auto_topk_mode,
        topk_max=topk_max,
        energy_threshold=energy_threshold,
        calib_samples=calib,
    )


# -----------------------
# Main conversion workflow
# -----------------------
def convert_to_fp8_super(
    input_file: str,
    output_file: str,
    t5xxl: bool = False,
    keep_distillation: bool = False,
    auto_topk_mode: str = "per-layer",  # {per-layer, global, off}
    manual_top_k: Optional[int] = None,
    num_iter: int = DEFAULT_NUM_ITER,
    calib_samples: int = DEFAULT_CALIB_SAMPLES,
    probe_k: int = DEFAULT_PROBE_K,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    topk_max: int = DEFAULT_TOPK_MAX,
):
    print(f"\n{'=' * 60}")
    print("FP8 Auto-Converter [TPEC-Quant]")
    print(f"{'=' * 60}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Mode:   {'T5XXL' if t5xxl else 'Flux/Chroma'}")
    print(f"{'=' * 60}\n")

    # Load model
    print("📂 Loading model...")
    tensors: Dict[str, torch.Tensor] = load_safetensors(input_file)
    print(f"✓ Loaded {len(tensors)} tensors\n")

    # Auto-detect T5 if not explicitly set
    auto_t5 = any((("decoder" in k) or ("lm_head" in k)) for k in tensors.keys())
    if auto_t5 and not t5xxl:
        print("ℹ️  Detected T5-like model; enabling T5-XXL handling automatically.")
        t5xxl = True
    print(f"T5-XXL mode: {t5xxl}\n")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine top_k strategy
    global_top_k = None
    if manual_top_k is not None:
        global_top_k = int(max(1, manual_top_k))
        print(f"🎯 Using manual global top_k: {global_top_k}\n")
    elif auto_topk_mode == "global":
        global_top_k = quick_inspect_model_global_topk(
            input_file, probe_k=probe_k, energy_threshold=energy_threshold, cap=topk_max
        )
        print()

    # Initialize converter
    print(f"⚙️  Initializing converter (num_iter={num_iter})...")
    converter = LearnedRoundingConverterTPEC(device=device, num_iter=num_iter)
    print()

    # Generate calibration data on device
    print("📊 Generating calibration data on device...")
    calib_cache: Dict[int, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if key.endswith(".weight") and tensor.ndim == 2 and tensor.numel() > 0:
            in_features = tensor.shape[1]
            if in_features not in calib_cache:
                calib_cache[in_features] = torch.randn(
                    calib_samples, in_features, dtype=COMPUTE_DTYPE, device=device
                )
    print(f"✓ Generated calibration data for {len(calib_cache)} layer sizes\n")

    # Process weights
    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith(".weight")])
    total_weights = len(weight_keys)
    processed_count = 0
    skipped_count = 0
    removed_count = 0

    print(f"🔄 Processing {total_weights} weight tensors...\n")
    for idx, key in enumerate(weight_keys, 1):
        W = tensors[key]
        base = key[: -len(".weight")]

        # T5 removal/skip
        if t5xxl and any(rm in key for rm in REMOVE_KEY_NAMES_T5):
            print(f"({idx}/{total_weights}) Removing T5 decoder tensor: {key}")
            removed_count += 1
            continue
        if t5xxl and any(av in key for av in AVOID_KEY_NAMES_T5):
            print(f"({idx}/{total_weights}) Skipping (T5 exclude): {key}")
            new_tensors[key] = W  # keep original
            skipped_count += 1
            continue

        # Keep distillation layers (optional)
        if keep_distillation and any(av in key for av in DISTILL_LAYER_KEYNAMES):
            print(f"({idx}/{total_weights}) Skipping (keep-distillation): {key}")
            new_tensors[key] = W
            new_tensors[f"{base}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            skipped_count += 1
            continue

        # Non-2D or empty weights: store FP8 cast + unit scale
        if W.ndim != 2 or W.numel() == 0:
            print(f"({idx}/{total_weights}) Non-2D/empty, keeping as-is FP8: {key}")
            new_tensors[key] = W.to(TARGET_FP8_DTYPE)
            new_tensors[f"{base}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        # Determine top_k for this tensor
        if global_top_k is not None:
            k_use = global_top_k
            print(f"({idx}/{total_weights}) Quantizing: {key}")
            print(f"    using global top_k={k_use}")
        elif auto_topk_mode == "per-layer":
            k_use, top1_frac = auto_recommend_top_k(
                W,
                device=device,
                probe_k=probe_k,
                energy_threshold=energy_threshold,
                topk_max=topk_max,
            )
            print(f"({idx}/{total_weights}) Quantizing: {key}")
            print(f"    recommended top_k={k_use} (top1_energy_frac={top1_frac:.3f})")
        else:  # off
            k_use = 1
            print(f"({idx}/{total_weights}) Quantizing: {key}")
            print(f"    auto top_k disabled, using top_k={k_use}")

        # Convert
        W_f8, dequant_scale, W_deq = converter.convert(
            W, top_k=k_use, topk_cap=topk_max
        )

        # Store results
        new_tensors[key] = W_f8
        new_tensors[f"{base}.scale_weight"] = dequant_scale.to(SCALE_DTYPE)

        # Bias correction (on device)
        bias_key = f"{base}.bias"
        if bias_key in tensors:
            # these are small matmuls; keep on device for speed
            W_orig_dev = W.to(device, dtype=COMPUTE_DTYPE)
            W_deq_dev = W_deq.to(device, dtype=COMPUTE_DTYPE)
            X_dev = calib_cache[W.shape[1]]
            b_dev = tensors[bias_key].to(device, dtype=COMPUTE_DTYPE)

            weight_error = W_orig_dev - W_deq_dev
            output_error = X_dev @ weight_error.T
            bias_correction = output_error.mean(dim=0)
            b_new = b_dev - bias_correction

            new_tensors[bias_key] = b_new.cpu().to(tensors[bias_key].dtype)

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
            # For T5 paths, also add scale_input (Comfy convention)
            new_tensors[f"{base}.scale_input"] = (
                dequant_scale.detach().clone().to(SCALE_DTYPE)
            )

        processed_count += 1

        # tidy memory
        del W_f8, dequant_scale, W_deq
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Add remaining tensors (skip removed ones in T5 mode)
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

    # Save
    print(f"\n💾 Saving {len(new_tensors)} tensors...")
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(new_tensors, output_file)

    print(f"\n{'=' * 60}")
    print("✅ Conversion Complete!")
    print(f"{'=' * 60}")
    print(f"Original tensors:  {len(tensors)}")
    print(f"Processed weights: {processed_count}")
    print(f"Skipped weights:   {skipped_count}")
    print(f"Removed (T5):      {removed_count}")
    print(f"Final tensors:     {len(new_tensors)}")
    print(f"Output saved to:   {output_file}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FP8 Scaled TPEC Converter (Smart Auto, per-layer/global auto top_k, presets, bias correction).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart Auto (no extra flags): choose quality/top_k/caps/calib by model and VRAM
  python fp8_tpec_conv.py --input model.safetensors

  # Global auto top_k via sampling
  python fp8_tpec_conv.py --input model.safetensors --auto-topk global

  # Manual global top_k
  python fp8_tpec_conv.py --input model.safetensors --top-k 2

  # T5-XXL mode (auto-detected if model has 'decoder'/'lm_head')
  python fp8_tpec_conv.py --input t5xxl.safetensors --t5xxl

  # Keep distillation layers unquantized
  python fp8_tpec_conv.py --input model.safetensors --keep-distillation

  # Quality presets
  python fp8_tpec_conv.py --input model.safetensors --quality fast
  python fp8_tpec_conv.py --input model.safetensors --quality high
""",
    )

    parser.add_argument(
        "--input", required=True, help="Input safetensors model (float weights)."
    )
    parser.add_argument(
        "--output", help="Output safetensors path (if omitted, auto-named)."
    )
    parser.add_argument("--t5xxl", action="store_true", help="Enable T5-XXL handling.")
    parser.add_argument(
        "--keep-distillation",
        action="store_true",
        help="Keep distillation layers unquantized (for Flux/Chroma variants).",
    )

    # Top-k controls
    parser.add_argument(
        "--auto-topk",
        choices=["per-layer", "global", "off"],
        default="per-layer",
        help="Auto top_k strategy: per-layer (default), global (sampled), or off (use 1 or manual).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Manual global top_k override (disables auto top_k).",
    )

    # Quality / perf knobs
    parser.add_argument(
        "--num-iter",
        type=int,
        default=DEFAULT_NUM_ITER,
        help="Optimization iterations (default: 500)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=DEFAULT_CALIB_SAMPLES,
        help="Calibration samples for bias correction (default: 3072)",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Preset quality level (adjusts num_iter & calib_samples).",
    )

    # Advanced auto-topk parameters
    parser.add_argument(
        "--probe-k",
        type=int,
        default=DEFAULT_PROBE_K,
        help="SVD probe rank (default: 8)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help="Energy capture threshold for SVD (default: 0.90)",
    )
    parser.add_argument(
        "--topk-max",
        type=int,
        default=DEFAULT_TOPK_MAX,
        help="Cap for top_k (default: 3)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return

    # Check FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except Exception:
        print("❌ Error: torch.float8_e4m3fn not supported by this PyTorch/hardware.")
        return

    # Generate output filename
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_fp8_tpec_scaled.safetensors"

    # Prevent overwriting input
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("❌ Error: Output file cannot be the same as input file")
        return

    # Early T5 detection for Smart Auto choice
    with safe_open(args.input, framework="pt", device="cpu") as f:
        head_keys = list(f.keys())[:64]
    auto_t5 = any((("decoder" in k) or ("lm_head" in k)) for k in head_keys)
    t5xxl_detected = args.t5xxl or auto_t5

    # Smart Auto: if user did not override core knobs, pick sensible defaults
    user_overrode_topk = (args.top_k is not None) or (args.auto_topk != "per-layer")
    user_overrode_quality = (
        (args.quality != "balanced")
        or (args.calib_samples != DEFAULT_CALIB_SAMPLES)
        or (args.num_iter != DEFAULT_NUM_ITER)
    )
    if not user_overrode_topk and not user_overrode_quality:
        auto = choose_auto_params(args.input, t5xxl_detected)
        print(f"🤖 Smart Auto profile: {auto}")
        args.quality = auto["quality"]
        args.auto_topk = auto["auto_topk_mode"]
        args.topk_max = auto["topk_max"]
        args.energy_threshold = auto["energy_threshold"]
        args.calib_samples = auto["calib_samples"]

    # Quality presets
    quality_presets = {
        "fast": {"num_iter": 250, "calib_samples": 1024},
        "balanced": {"num_iter": 500, "calib_samples": 3072},
        "high": {"num_iter": 1000, "calib_samples": 8192},
    }
    if args.quality in quality_presets:
        preset = quality_presets[args.quality]
        num_iter = (
            args.num_iter if args.num_iter != DEFAULT_NUM_ITER else preset["num_iter"]
        )
        calib_samples = (
            args.calib_samples
            if args.calib_samples != DEFAULT_CALIB_SAMPLES
            else preset["calib_samples"]
        )
    else:
        num_iter = args.num_iter
        calib_samples = args.calib_samples

    # If manual top_k provided, force auto_topk off
    auto_topk_mode = args.auto_topk
    if args.top_k is not None:
        auto_topk_mode = "off"

    # Run conversion with OOM resilience
    try:
        convert_to_fp8_super(
            input_file=args.input,
            output_file=args.output,
            t5xxl=t5xxl_detected,
            keep_distillation=args.keep_distillation,
            auto_topk_mode=auto_topk_mode,
            manual_top_k=args.top_k,
            num_iter=num_iter,
            calib_samples=calib_samples,
            probe_k=args.probe_k,
            energy_threshold=args.energy_threshold,
            topk_max=args.topk_max,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️  OOM detected, retrying with smaller calibration and caps...")
            # Degrade settings and retry once
            args.calib_samples = max(1024, calib_samples // 2)
            args.topk_max = max(1, min(args.topk_max, 2))
            auto_topk_mode = "global" if auto_topk_mode == "per-layer" else "off"
            convert_to_fp8_super(
                input_file=args.input,
                output_file=args.output,
                t5xxl=t5xxl_detected,
                keep_distillation=args.keep_distillation,
                auto_topk_mode=auto_topk_mode,
                manual_top_k=args.top_k,
                num_iter=max(250, num_iter // 2),
                calib_samples=args.calib_samples,
                probe_k=args.probe_k,
                energy_threshold=max(0.85, args.energy_threshold - 0.05),
                topk_max=args.topk_max,
            )
        else:
            raise


if __name__ == "__main__":
    main()
