#!/usr/bin/env python3
"""
selective_requant_from_inspector.py

Workflow:
  1) Run svd_inspector_with_json.py to create a JSON report (e.g. svd_report.json).
  2) Run convert_fp8_scaled_learned_svd_fast.py once with a quick/default top_k (e.g. 1)
     to produce an initial quantized safetensors file (initial_quant.safetensors).
  3) Run this script to selectively re-quantize only tensors flagged by the inspector
     with recommended_top_k > baseline_top_k, and merge replacements into a new safetensors file.

Requirements / assumptions:
  - Run this script from the repository root of the Learned-Rounding project (so the
    conversion module convert_fp8_scaled_learned_svd_fast.py is importable).
  - Python environment has: torch, safetensors, svd_inspector_with_json.py output JSON.
  - You provided:
      * original model (float32) safetensors
      * initial quantized safetensors (produced by learned_svd_fast with baseline top_k)
      * inspector JSON from svd_inspector_with_json.py
  - This script imports LearnedRoundingConverter from convert_fp8_scaled_learned_svd_fast.py.
    If you've moved or renamed that file, update the import.

What it does:
  - Reads inspector JSON and finds tensors where recommended_top_k > baseline_top_k.
  - For each such tensor, re-runs the learned-SVD conversion with the suggested top_k and num_iter.
  - Replaces the quantized tensor and scale entries in the initial quantized safetensors file.
  - Optionally updates the bias (bias correction) consistent with the converter's bias-correction logic.
  - Writes a merged output safetensors file.

Usage (example):
  python selective_requant_from_inspector.py \
    --orig /path/to/original_model.safetensors \
    --initial /path/to/initial_quant.safetensors \
    --inspector svd_report.json \
    --out /path/to/quant_refined.safetensors \
    --baseline-topk 1 \
    --num-iter 500 \
    --calib-samples 3072

Notes:
  - This is a pragmatic bridge-tool: it avoids re-quantizing the entire model and focuses time
    on the tensors that the inspector recommends increasing top_k for.
  - You can also set --force-all to re-quantize everything according to inspector recommendations.
"""

import argparse
import json
import os
import sys
import gc
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Import the LearnedRoundingConverter from the repo. Run from repo root.
try:
    from convert_fp8_scaled_learned_svd_fast import LearnedRoundingConverter
    # If desired, pull TARGET_FP8_DTYPE and SCALE_DTYPE names for dtype agreement:
    try:
        TARGET_FP8_DTYPE = getattr(__import__("convert_fp8_scaled_learned_svd_fast"), "TARGET_FP8_DTYPE")
    except Exception:
        TARGET_FP8_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None
except Exception as e:
    print("ERROR: Could not import LearnedRoundingConverter from convert_fp8_scaled_learned_svd_fast.py.")
    print("Make sure you run this script from the Learned-Rounding repo root and that the file exists.")
    print("Import error:", e)
    sys.exit(2)


def load_safetensors_to_dict(path: str) -> Dict[str, torch.Tensor]:
    d = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            d[k] = f.get_tensor(k).cpu()
    return d


def main():
    p = argparse.ArgumentParser(description="Selective re-quantization using inspector recommendations")
    p.add_argument("--orig", required=True, help="Original float safetensors (source of truth weights)")
    p.add_argument("--initial", required=True, help="Initial quantized safetensors (baseline run output)")
    p.add_argument("--inspector", required=True, help="JSON report from svd_inspector_with_json.py")
    p.add_argument("--out", required=True, help="Output safetensors path to write merged/updated model")
    p.add_argument("--baseline-topk", type=int, default=1, help="Top_k used for the initial quant run")
    p.add_argument("--num-iter", type=int, default=500, help="num_iter to use for re-quantizing flagged tensors")
    p.add_argument("--calib-samples", type=int, default=3072, help="Calibration samples used for bias correction")
    p.add_argument("--force-all", action="store_true", help="Ignore inspector filtering and re-quantize all 2D weight tensors using recommended_top_k")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto", help="Device to run converter on (auto=prefer cuda if available)")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    print(f"Running on device: {device}")

    print("Loading inspector JSON...")
    with open(args.inspector, "r", encoding="utf-8") as jf:
        inspector = json.load(jf)

    # Map recommended top_k by tensor key
    rec_map = {entry["key"]: int(entry.get("recommended_top_k", 1)) for entry in inspector}

    print("Loading original model (float)...")
    orig = load_safetensors_to_dict(args.orig)
    print(f"Loaded {len(orig)} tensors from original model.")

    print("Loading initial quantized model...")
    initial = load_safetensors_to_dict(args.initial)
    print(f"Loaded {len(initial)} tensors from initial quantized model.")

    # Output dict starts as a copy of initial quantized tensors
    new_tensors = {k: v.clone() for k, v in initial.items()}

    # Instantiate a converter (we'll re-create with different top_k per-tensor)
    # We'll only import/instantiate the class as needed to set different top_k.
    processed = 0
    skipped = 0
    replaced = 0
    failed = 0

    weight_keys = [k for k in orig.keys() if k.endswith(".weight") and orig[k].ndim == 2 and orig[k].numel() > 0]
    print(f"Found {len(weight_keys)} 2D weight tensors in original model.")

    for key in weight_keys:
        recommended = rec_map.get(key, args.baseline_topk)
        if args.force_all:
            pass  # we will re-quantize according to recommended (or default if missing)
        else:
            if recommended <= args.baseline_topk:
                skipped += 1
                continue

        print(f"\nRe-quantizing tensor: {key} -> recommended top_k = {recommended}")
        W_orig = orig[key].to(device=device, dtype=torch.float32)

        # Create calibration data as random tensors (same heuristic as converter)
        in_features = W_orig.shape[1]
        X_calib = torch.randn(args.calib_samples, in_features, dtype=torch.float32, device=device)

        # instantiate converter with per-tensor top_k and num_iter
        converter = LearnedRoundingConverter(num_iter=args.num_iter, top_k=recommended)
        # ensure converter uses selected device
        try:
            converter.device = device
        except Exception:
            pass

        try:
            W_f8, dequant_scale, W_dequant = converter.convert(W_orig.cpu() if converter.device == "cpu" else W_orig, X_calib.cpu() if converter.device == "cpu" else X_calib)
            # convert return values are on CPU per the converter code - but be resilient
            # Place results into new_tensors (store FP8 quantized tensor and scale)
            new_tensors[key] = W_f8.clone().cpu()
            base_name = key[:-len(".weight")]
            scale_key = f"{base_name}.scale_weight"
            new_tensors[scale_key] = dequant_scale.clone().to(torch.float32)

            # Bias correction: replicate the converter's bias-correction logic
            bias_key = f"{base_name}.bias"
            if bias_key in orig:
                print(f"  - Adjusting bias for: {bias_key}")
                # do bias correction on device (use compute dtype float32)
                device_for_corr = "cuda" if torch.cuda.is_available() else "cpu"
                W_orig_dev = W_orig.to(device_for_corr, dtype=torch.float32)
                W_dequant_dev = W_dequant.to(device_for_corr, dtype=torch.float32)
                X_dev = X_calib.to(device_for_corr, dtype=torch.float32)
                b_orig_dev = orig[bias_key].to(device_for_corr, dtype=torch.float32)

                weight_error = W_orig_dev - W_dequant_dev.to(device_for_corr, dtype=torch.float32)
                output_error = X_dev @ weight_error.T
                bias_correction = output_error.mean(dim=0)
                b_new = b_orig_dev - bias_correction

                new_tensors[bias_key] = b_new.cpu().to(orig[bias_key].dtype)
                # free
                del W_orig_dev, W_dequant_dev, X_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            replaced += 1
            processed += 1
            # cleanup between tensors
            del W_f8, dequant_scale, W_dequant
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: failed to re-quantize {key}: {e}")
            failed += 1
            # don't crash whole run; continue to next tensor
            continue

    # Add any original non-quantized tensors that were missing from initial quant file
    for k, t in orig.items():
        if k not in new_tensors:
            new_tensors[k] = t.clone()
            # if needed, copy scale keys for matching weight entries
            # (we only replaced some weight entries; keep other tensors identical)

    # Save merged safetensors
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving merged quantized model to: {args.out} ...")
    try:
        save_file(new_tensors, args.out)
        print("Saved.")
    except Exception as e:
        print("ERROR saving output safetensors:", e)
        sys.exit(3)

    print("\nSummary:")
    print(f"  processed (attempted): {processed + skipped}")
    print(f"  replaced tensors     : {replaced}")
    print(f"  skipped (baseline ok): {skipped}")
    print(f"  failed               : {failed}")
    print("Done.")


if __name__ == "__main__":
    main()