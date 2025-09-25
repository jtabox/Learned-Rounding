#!/usr/bin/env python3
"""
one_shot_learned_svd_quant.py

One-shot workflow to inspect a safetensors model, choose per-tensor top_k recommendations,
run a baseline learned-SVD quantization pass, then selectively re-quantize flagged tensors,
and write a final quantized safetensors file.

Usage (example):
  python one_shot_learned_svd_quant.py \
    --input /path/to/orig.safetensors \
    --output /path/to/final_quant.safetensors \
    --baseline-topk 1 --num-iter 500 --probe-k 8 --energy-threshold 0.90 --calib-samples 3072

Notes:
- This script expects convert_fp8_scaled_learned_svd_fast.py (LearnedRoundingConverter class)
  to be present and importable from the current working directory (repo root).
- Requires torch and safetensors.
- It tries to use CUDA if available. Large models will need sufficient GPU memory.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import gc
from typing import List, Tuple, Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- Embedded inspector helpers (adapted from earlier svd_inspector_with_json) ---
def compute_top_singulars(
    W: torch.Tensor, k: int, use_fast: bool = True, pca_niter: int = 8
) -> torch.Tensor:
    m, n = W.shape
    k = min(k, m, n)
    if k <= 0:
        return torch.tensor([], dtype=torch.float32)
    try:
        if use_fast and min(m, n) > 256 and k < min(m, n) // 2:
            U, S, V = torch.pca_lowrank(W, q=k, center=False, niter=pca_niter)
            return S.cpu()
        else:
            S_all = torch.linalg.svdvals(W)
            return S_all[:k].cpu()
    except Exception:
        S_all = torch.linalg.svdvals(W)
        return S_all[:k].cpu()


def recommend_top_k_from_singulars(
    singulars: torch.Tensor, total_energy: float, energy_threshold: float = 0.90
) -> Tuple[int, bool]:
    if singulars.numel() == 0 or total_energy <= 0:
        return 1, False
    sigma_sq = singulars**2
    cum = torch.cumsum(sigma_sq, dim=0).cpu().numpy()
    thresh = energy_threshold * total_energy
    idxs = [i for i, v in enumerate(cum) if v >= thresh]
    if len(idxs) == 0:
        return int(len(singulars)), True
    return int(idxs[0] + 1), (idxs[0] + 1) == len(singulars)


def inspect_model(
    path: str,
    top_k_probe: int = 8,
    energy_threshold: float = 0.90,
    pca_niter: int = 8,
    verbose: bool = False,
) -> List[dict]:
    report = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            if not key.endswith(".weight"):
                continue
            W = f.get_tensor(key).to(torch.float32).cpu()
            if W.ndim != 2 or W.numel() == 0:
                continue
            m, n = W.shape
            k_probe = min(top_k_probe, min(m, n))
            total_energy = float(torch.linalg.norm(W, ord="fro") ** 2)
            if total_energy <= 0.0 or torch.allclose(W, torch.zeros_like(W)):
                entry = {
                    "key": key,
                    "shape": [m, n],
                    "note": "zero",
                    "topk_probed": 0,
                    "top_singulars": [],
                    "topk_energy_frac": 0.0,
                    "top1_energy_frac": 0.0,
                    "recommended_top_k": 1,
                    "probe_limited": False,
                }
                report.append(entry)
                continue
            S = compute_top_singulars(W, k_probe, use_fast=True, pca_niter=pca_niter)
            sigma_sq = (S**2) if S.numel() > 0 else torch.tensor([])
            topk_energy = float(sigma_sq.sum().item()) if sigma_sq.numel() > 0 else 0.0
            top1_energy = float(sigma_sq[0].item()) if sigma_sq.numel() > 0 else 0.0
            topk_energy_frac = topk_energy / total_energy if total_energy > 0 else 0.0
            top1_energy_frac = top1_energy / total_energy if total_energy > 0 else 0.0
            recommended_k, probe_limited = recommend_top_k_from_singulars(
                S, total_energy, energy_threshold=energy_threshold
            )
            cond_est = None
            if S.numel() > 1 and S[-1].item() > 0:
                cond_est = float((S[0] / S[-1]).item())
            entry = {
                "key": key,
                "shape": [m, n],
                "topk_probed": int(k_probe),
                "top_singulars": [float(x) for x in S.tolist()],
                "topk_energy_frac": topk_energy_frac,
                "top1_energy_frac": top1_energy_frac,
                "total_energy_estimated": total_energy,
                "recommended_top_k": int(recommended_k),
                "probe_limited": bool(probe_limited),
                "condition_estimate_from_probed": cond_est,
            }
            report.append(entry)
            if verbose:
                print(
                    f"inspector: {key} shape={m}x{n} top1_frac={top1_energy_frac:.3f} -> rec={recommended_k}{' (probe-limited)' if probe_limited else ''}"
                )
    return report


# --- Helper to load a safetensors file into a dict ---
def load_safetensors_dict(path: str) -> Dict[str, torch.Tensor]:
    d = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            d[k] = f.get_tensor(k).cpu()
    return d


# --- Try import converter class from repo file(s) ---
def import_converter_class():
    possible_modules = [
        "convert_fp8_scaled_learned_svd_fast",
        "convert_fp8_scaled_learned_svd",
        "convert_fp8_scaled_learned",
    ]
    for modname in possible_modules:
        try:
            mod = __import__(modname)
            if hasattr(mod, "LearnedRoundingConverter"):
                return mod.LearnedRoundingConverter, getattr(
                    mod, "TARGET_FP8_DTYPE", None
                )
        except Exception:
            continue
    return None, None


# --- Main orchestration ---
def main():
    p = argparse.ArgumentParser(
        description="One-shot inspect + learned-SVD quantize workflow"
    )
    p.add_argument(
        "--input", required=True, help="Input original safetensors model (float)"
    )
    p.add_argument("--output", required=True, help="Output quantized safetensors path")
    p.add_argument(
        "--baseline-topk", type=int, default=1, help="Baseline top_k for initial pass"
    )
    p.add_argument(
        "--num-iter", type=int, default=500, help="num_iter used in converter"
    )
    p.add_argument(
        "--calib-samples",
        type=int,
        default=3072,
        help="Calibration samples for bias correction",
    )
    p.add_argument(
        "--probe-k", type=int, default=8, help="How many top singular values to probe"
    )
    p.add_argument(
        "--energy-threshold",
        type=float,
        default=0.90,
        help="Energy capture threshold for recommending top_k",
    )
    p.add_argument(
        "--pca-niter", type=int, default=8, help="pca_lowrank iterations for inspector"
    )
    p.add_argument(
        "--force-all",
        action="store_true",
        help="Ignore inspector filtering and re-quantize all tensors with recommended_top_k",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose prints")
    args = p.parse_args()

    # device selection
    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        or args.device == "cuda"
        else "cpu"
    )
    print(f"Device: {device}")

    # Import converter
    ConverterClass, TARGET_FP8_DTYPE = import_converter_class()
    if ConverterClass is None:
        print(
            "ERROR: Could not find LearnedRoundingConverter in repo. Make sure you run this script from the Learned-Rounding repo root."
        )
        sys.exit(2)

    print(
        "Running inspector to probe singular-value decay (this is fast relative to quantization)..."
    )
    inspector = inspect_model(
        args.input,
        top_k_probe=args.probe_k,
        energy_threshold=args.energy_threshold,
        pca_niter=args.pca_niter,
        verbose=args.verbose,
    )
    # map recommendations
    rec_map = {
        entry["key"]: int(entry.get("recommended_top_k", 1)) for entry in inspector
    }

    # Prepare calibration data cache similar to repo behavior
    print("Loading model tensors (CPU) to enumerate shapes...")
    orig_tensors = load_safetensors_dict(args.input)
    calib_cache: Dict[int, torch.Tensor] = {}
    for k, t in orig_tensors.items():
        if k.endswith(".weight") and t.ndim == 2:
            in_features = t.shape[1]
            if in_features not in calib_cache:
                calib_cache[in_features] = torch.randn(
                    args.calib_samples, in_features, dtype=torch.float32
                )

    # Initial baseline quantization pass (per-tensor converter.convert)
    print("Starting initial baseline quantization pass...")
    converter_baseline = ConverterClass(
        num_iter=args.num_iter, top_k=args.baseline_topk
    )
    try:
        converter_baseline.device = device
    except Exception:
        pass

    new_tensors: Dict[str, torch.Tensor] = {}
    processed = 0
    for key, tensor in orig_tensors.items():
        if not key.endswith(".weight") or tensor.ndim != 2 or tensor.numel() == 0:
            continue
        processed += 1
        print(f"[baseline] Processing: {key} ({processed})")
        W_orig = tensor.to(torch.float32)
        in_features = W_orig.shape[1]
        X_calib = calib_cache[in_features].to(
            device if converter_baseline.device == "cuda" else "cpu"
        )
        try:
            W_f8, dequant_scale, dequantized = converter_baseline.convert(
                W_orig.to(converter_baseline.device), X_calib
            )
            new_tensors[key] = W_f8.clone().cpu()
            base = key[: -len(".weight")]
            new_tensors[f"{base}.scale_weight"] = dequant_scale.clone().to(
                torch.float32
            )
            # bias correction if bias exists
            bias_key = f"{base}.bias"
            if bias_key in orig_tensors:
                print(f"  - bias correction: {bias_key}")
                device_corr = "cuda" if torch.cuda.is_available() else "cpu"
                W_orig_dev = W_orig.to(device_corr)
                W_deq_dev = dequantized.to(device_corr)
                X_dev = X_calib.to(device_corr)
                b_orig_dev = orig_tensors[bias_key].to(device_corr, dtype=torch.float32)
                weight_error = W_orig_dev - W_deq_dev.to(device_corr)
                output_error = X_dev @ weight_error.T
                bias_correction = output_error.mean(dim=0)
                b_new = b_orig_dev - bias_correction
                new_tensors[bias_key] = b_new.cpu().to(orig_tensors[bias_key].dtype)
                # cleanup
                del (
                    W_orig_dev,
                    W_deq_dev,
                    X_dev,
                    b_orig_dev,
                    weight_error,
                    output_error,
                    bias_correction,
                    b_new,
                )
                if device_corr == "cuda":
                    torch.cuda.empty_cache()
            # cleanup converter outputs
            del W_f8, dequant_scale, dequantized
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR baseline quant for {key}: {e}")
            # fallback: store original (or quantize with naive RtN)
            new_tensors[key] = tensor.to(torch.float32)
            base = key[: -len(".weight")]
            new_tensors[f"{base}.scale_weight"] = torch.tensor(
                [1.0], dtype=torch.float32
            )

    # Add non-weight tensors (metadata etc.) from original that weren't replaced (or baseline skipped)
    for k, t in orig_tensors.items():
        if k not in new_tensors:
            new_tensors[k] = t.clone()

    # Selective re-quantization pass
    print("Selective re-quantization: scanning inspector recommendations...")
    replaced = 0
    skipped = 0
    failed = 0
    for entry in inspector:
        key = entry["key"]
        rec_k = int(entry.get("recommended_top_k", 1))
        if key not in orig_tensors:
            continue
        if not args.force_all and rec_k <= args.baseline_topk:
            skipped += 1
            continue
        print(f"[re-quant] Re-quantizing {key} with top_k={rec_k}")
        W_orig = orig_tensors[key].to(torch.float32)
        in_features = W_orig.shape[1]
        X_calib = calib_cache[in_features].to(device)
        # instantiate converter per recommended top_k
        Converter = ConverterClass(num_iter=args.num_iter, top_k=rec_k)
        try:
            Converter.device = device
        except Exception:
            pass
        try:
            W_f8, dequant_scale, dequantized = Converter.convert(
                W_orig.to(Converter.device),
                X_calib if Converter.device == "cuda" else X_calib.cpu(),
            )
            new_tensors[key] = W_f8.clone().cpu()
            base = key[: -len(".weight")]
            new_tensors[f"{base}.scale_weight"] = dequant_scale.clone().to(
                torch.float32
            )
            # bias correction
            bias_key = f"{base}.bias"
            if bias_key in orig_tensors:
                print(f"  - bias correction: {bias_key}")
                device_corr = "cuda" if torch.cuda.is_available() else "cpu"
                W_orig_dev = W_orig.to(device_corr)
                W_deq_dev = dequantized.to(device_corr)
                X_dev = X_calib.to(device_corr)
                b_orig_dev = orig_tensors[bias_key].to(device_corr, dtype=torch.float32)
                weight_error = W_orig_dev - W_deq_dev.to(device_corr)
                output_error = X_dev @ weight_error.T
                bias_correction = output_error.mean(dim=0)
                b_new = b_orig_dev - bias_correction
                new_tensors[bias_key] = b_new.cpu().to(orig_tensors[bias_key].dtype)
                del (
                    W_orig_dev,
                    W_deq_dev,
                    X_dev,
                    b_orig_dev,
                    weight_error,
                    output_error,
                    bias_correction,
                    b_new,
                )
                if device_corr == "cuda":
                    torch.cuda.empty_cache()
            replaced += 1
            del W_f8, dequant_scale, dequantized
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR re-quant for {key}: {e}")
            failed += 1
            continue

    # Save final merged safetensors
    print(f"Saving final quantized model to: {args.output}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        save_file(new_tensors, args.output)
        print("Saved final quantized model.")
    except Exception as e:
        print("ERROR saving:", e)
        sys.exit(3)

    print("\nSummary:")
    print(f"  baseline processed weights : {processed}")
    print(f"  selective replaced         : {replaced}")
    print(f"  selective skipped          : {skipped}")
    print(f"  selective failed           : {failed}")
    print("Done.")


if __name__ == "__main__":
    main()
