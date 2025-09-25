#!/usr/bin/env python3
"""
svd_inspector_with_json.py

Inspect singular-value decay of 2-D weight tensors in a safetensors model and
produce a JSON report listing per-tensor metrics and a recommended top_k.

Usage:
    python svd_inspector_with_json.py --model /path/to/model.safetensors \
        --probe-k 8 --energy-threshold 0.90 --output report.json --verbose

The script:
- Probes up to --probe-k top singular values per 2D weight tensor (uses fast PCA for large matrices).
- Computes energy fractions using singular values^2 (so energy = sum(sigma_i^2) equals Frobenius norm^2).
- Recommends a smallest top_k (within probed k) that captures >= --energy-threshold of total energy.
- Writes a JSON array with entries for each weight tensor including metrics and recommendation.
"""

import argparse
import json
import math
from typing import List, Tuple

import torch
from safetensors import safe_open


def compute_top_singulars(
    W: torch.Tensor, k: int, use_fast: bool = True, pca_niter: int = 8
) -> torch.Tensor:
    """
    Return the top-k singular values of W (descending).
    Uses torch.pca_lowrank for large matrices when use_fast=True.
    W should be float32 on CPU.
    """
    m, n = W.shape
    k = min(k, m, n)
    if k <= 0:
        return torch.tensor([], dtype=torch.float32)
    try:
        if use_fast and min(m, n) > 256 and k < min(m, n) // 2:
            # U, S, V returned by pca_lowrank -> S are singular values for the top components
            U, S, V = torch.pca_lowrank(W, q=k, center=False, niter=pca_niter)
            return S.cpu()
        else:
            # compute full SVD values and slice
            S_all = torch.linalg.svdvals(W)
            return S_all[:k].cpu()
    except Exception:
        # fallback to full SVD if pca_lowrank fails
        S_all = torch.linalg.svdvals(W)
        return S_all[:k].cpu()


def recommend_top_k_from_singulars(
    singulars: torch.Tensor, total_energy: float, energy_threshold: float = 0.90
) -> Tuple[int, bool]:
    """
    Given probed singular values (descending) and the full-matrix total energy (sum of sigma_i^2),
    find the smallest k (<=len(singulars)) such that sum_{i<=k} sigma_i^2 >= energy_threshold * total_energy.

    Returns (recommended_k, probe_limited)
    - probe_limited = True if recommendation equals len(singulars) (might need a larger probe-k).
    """
    if singulars.numel() == 0 or total_energy <= 0:
        return 1, False
    sigma_sq = singulars**2
    cum = torch.cumsum(sigma_sq, dim=0).cpu().numpy()
    thresh = energy_threshold * total_energy
    idxs = [i for i, v in enumerate(cum) if v >= thresh]
    if len(idxs) == 0:
        # even all probed singulars don't reach threshold — recommend probe limit and mark limited
        return int(len(singulars)), True
    return int(idxs[0] + 1), (idxs[0] + 1) == len(singulars)


def inspect_model_and_report(
    path: str,
    top_k_probe: int = 8,
    energy_threshold: float = 0.90,
    verbose: bool = False,
    output_json: str = "svd_report.json",
    pca_niter: int = 8,
) -> List[dict]:
    """
    Inspect the safetensors model and write a JSON report with per-weight recommendations.
    Returns the in-memory list of per-tensor dicts (same data as written to JSON).
    """
    report = []
    print(f"Inspecting model: {path}")
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

            # total energy (sum of squared singular values == Frobenius norm^2)
            total_energy = float(torch.linalg.norm(W, ord="fro") ** 2)

            # If matrix is nearly zero, skip heavy ops
            if total_energy <= 0.0 or torch.allclose(W, torch.zeros_like(W)):
                entry = {
                    "key": key,
                    "shape": [m, n],
                    "note": "all-zero or negligible",
                    "topk_probed": 0,
                    "top_singulars": [],
                    "top1_energy_frac": 0.0,
                    "topk_energy_frac": 0.0,
                    "recommended_top_k": 1,
                    "probe_limited": False,
                }
                report.append(entry)
                if verbose:
                    print(f"- {key} shape={m}x{n} is all-zero or negligible; skipping.")
                continue

            S = compute_top_singulars(W, k_probe, use_fast=True, pca_niter=pca_niter)
            S_list = [float(x) for x in S.tolist()]
            # compute energy fractions using squared singular values
            sigma_sq = (S**2) if S.numel() > 0 else torch.tensor([])
            topk_energy = float(sigma_sq.sum().item()) if sigma_sq.numel() > 0 else 0.0
            top1_energy = float(sigma_sq[0].item()) if sigma_sq.numel() > 0 else 0.0

            topk_energy_frac = topk_energy / total_energy if total_energy > 0 else 0.0
            top1_energy_frac = top1_energy / total_energy if total_energy > 0 else 0.0

            recommended_k, probe_limited = recommend_top_k_from_singulars(
                S, total_energy, energy_threshold=energy_threshold
            )

            # condition estimate (from probed singulars) — if only one probed value, set None/inf
            cond_est = None
            if S.numel() > 1 and S[-1].item() > 0:
                cond_est = float((S[0] / S[-1]).item())

            entry = {
                "key": key,
                "shape": [m, n],
                "topk_probed": int(k_probe),
                "top_singulars": S_list,
                "topk_energy_frac": topk_energy_frac,
                "top1_energy_frac": top1_energy_frac,
                "total_energy_estimated": total_energy,
                "recommended_top_k": int(recommended_k),
                "probe_limited": bool(probe_limited),
                "condition_estimate_from_probed": cond_est,
            }

            report.append(entry)

            # console output for quick scanning
            print(
                f"- {key} | shape={m}x{n} | top1_energy_frac={top1_energy_frac:.3f} | top{len(S)}_energy_frac={topk_energy_frac:.3f} | cond_est={cond_est if cond_est is not None else 'N/A'} -> recommend top_k={recommended_k}{' (probe-limited)' if probe_limited else ''}"
            )
            if verbose:
                if S.numel() > 0:
                    cum_energy = (torch.cumsum(S**2, dim=0) / total_energy).tolist()
                    print(
                        f"    singulars (top{len(S)}): "
                        + ", ".join(f"{x:.4e}" for x in S_list)
                    )
                    print(
                        "    cumulative energy (probed): "
                        + ", ".join(f"{x:.3f}" for x in cum_energy)
                    )

    # write JSON report
    try:
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        print(f"\nWrote JSON report to: {output_json} (entries: {len(report)})")
    except Exception as e:
        print(f"Error writing JSON report to '{output_json}': {e}")

    return report


def main():
    p = argparse.ArgumentParser(
        description="Inspect singular-value decay and recommend top_k for learned_svd_fast"
    )
    p.add_argument("--model", required=True, help="Path to safetensors model")
    p.add_argument(
        "--probe-k",
        type=int,
        default=8,
        help="How many top singular values to probe per tensor",
    )
    p.add_argument(
        "--energy-threshold",
        type=float,
        default=0.90,
        help="Cumulative-energy threshold for recommending top_k (0-1)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full top-k singular values and cumulative energy",
    )
    p.add_argument(
        "--output", type=str, default="svd_report.json", help="Output JSON report path"
    )
    p.add_argument(
        "--pca-niter",
        type=int,
        default=8,
        help="Number of power iterations for pca_lowrank (if used)",
    )
    args = p.parse_args()

    inspect_model_and_report(
        args.model,
        top_k_probe=args.probe_k,
        energy_threshold=args.energy_threshold,
        verbose=args.verbose,
        output_json=args.output,
        pca_niter=args.pca_niter,
    )


if __name__ == "__main__":
    main()
