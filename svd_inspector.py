# svd_inspector.py
import argparse
import torch
from safetensors import safe_open
import math


def compute_top_singulars(W: torch.Tensor, k: int, use_fast: bool = True):
    # W expected on CPU, float32
    m, n = W.shape
    k = min(k, m, n)
    if k == 0:
        return torch.tensor([])
    # Use pca_lowrank for large matrices (faster for top-k), fallback to svdvals for small
    if use_fast and min(m, n) > 256 and k < min(m, n) // 2:
        U, S, V = torch.pca_lowrank(W, q=k, center=False, niter=8)
        return S
    else:
        # returns all singular values (descending)
        S_all = torch.linalg.svdvals(W)
        return S_all[:k]


def recommend_top_k(singulars: torch.Tensor, energy_threshold: float = 0.90):
    if singulars.numel() == 0:
        return 1
    total = singulars.sum().item()
    if total <= 0:
        return 1
    cum = torch.cumsum(singulars, dim=0) / total
    # find smallest k where cumulative >= threshold
    kvals = (cum >= energy_threshold).nonzero(as_tuple=False)
    if kvals.numel() == 0:
        return len(singulars)
    return int(kvals[0].item() + 1)


def inspect_model(
    path: str,
    top_k_probe: int = 8,
    energy_threshold: float = 0.90,
    verbose: bool = False,
):
    print(f"Inspecting: {path}")
    stats = []
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
            S = compute_top_singulars(W, k_probe, use_fast=True)
            if S.numel() == 0:
                continue
            s0 = S[0].item()
            ssum = S.sum().item()
            top1_frac = s0 / ssum if ssum > 0 else 0.0
            # estimate condition number from available singulars (rough)
            cond_est = (
                (S[0] / S[-1]).item()
                if S.numel() > 1 and S[-1].item() > 0
                else float("inf")
            )
            recommended_k = recommend_top_k(S, energy_threshold=energy_threshold)

            print(
                f"- {key} | shape={m}x{n} | top1_frac={top1_frac:.3f} | top{len(S)}_energy={ssum:.3e} (sum of probed svs) | cond_est={cond_est:.3f} -> recommend top_k={recommended_k}"
            )
            if verbose:
                cum = (torch.cumsum(S, dim=0) / ssum).tolist() if ssum > 0 else []
                print(
                    f"    singulars (top{len(S)}): "
                    + ", ".join(f"{x:.4e}" for x in S.tolist())
                )
                print("    cumulative energy: " + ", ".join(f"{x:.3f}" for x in cum))
            stats.append((key, m, n, top1_frac, recommended_k))
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inspect singular-value decay of 2D weight tensors in a safetensors model"
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
        help="Cumulative-energy threshold for recommending top_k",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print full top-k singular values"
    )
    args = p.parse_args()
    inspect_model(
        args.model,
        top_k_probe=args.probe_k,
        energy_threshold=args.energy_threshold,
        verbose=args.verbose,
    )
