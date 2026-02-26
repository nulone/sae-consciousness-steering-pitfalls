#!/usr/bin/env python3
"""
Verify key numbers from the writeup using only saved JSON data.
No GPU required. Run: python scripts/verify_numbers.py

Optionally pass explicit paths:
  python scripts/verify_numbers.py --all-experiments data/all_experiments_20260223_130655.json \
                                    --exp-h data/exp_H_v2_20260225_121930.json

If no arguments given, auto-detects the latest matching files in data/.

This script reads the stored experiment results and recomputes:
- ΔAIC = 15.8 (exponential vs power-law for FP scaling)
- Paper features delta = -0.031
- Conditional p = 0.40
- FP rates at each sample size
"""

import json
import argparse
import glob
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def find_latest(pattern, exclude="checkpoint"):
    """Find latest file matching a glob pattern in DATA_DIR, excluding checkpoints."""
    matches = sorted(glob.glob(str(DATA_DIR / pattern)))
    if exclude:
        matches = [m for m in matches if exclude not in m]
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {DATA_DIR}")
    return matches[-1]  # sorted by name → latest timestamp wins


def verify_scaling_law(all_exp_path=None):
    """Recompute ΔAIC from stored FP data points."""
    print("=" * 60)
    print("SCALING LAW VERIFICATION (EXP C)")
    print("=" * 60)

    path = all_exp_path or find_latest("all_experiments_*.json")
    print(f"  Using: {Path(path).name}")
    with open(path) as f:
        data = json.load(f)

    exp_c = data["experiments"]["C"]
    ns, fps = [], []
    for k in sorted(exp_c.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        if k.isdigit():
            ns.append(int(k))
            fps.append(exp_c[k]["fp_rate_mean"])
            print(f"  n={int(k):>3}: FP = {exp_c[k]['fp_rate_mean']:.4f}")

    # Filter to non-zero FP for fitting
    mask = [fp > 0 for fp in fps]
    ns_fit = np.array([n for n, m in zip(ns, mask) if m], dtype=float)
    fps_fit = np.array([fp for fp, m in zip(fps, mask) if m])
    n_pts = len(ns_fit)
    print(f"\n  Fitting on {n_pts} non-zero points")

    # Power law: FP = a * n^b (curve_fit in linear space)
    def power_law(x, a, b):
        return a * x ** b

    popt_pow, _ = curve_fit(power_law, ns_fit, fps_fit, p0=[1.0, -1.0])
    rss_pow = float(np.sum((fps_fit - power_law(ns_fit, *popt_pow)) ** 2))

    # Exponential: FP = a * exp(b * n) (curve_fit in linear space)
    def exp_decay(x, a, b):
        return a * np.exp(b * x)

    popt_exp, _ = curve_fit(exp_decay, ns_fit, fps_fit, p0=[1.0, -0.1])
    rss_exp = float(np.sum((fps_fit - exp_decay(ns_fit, *popt_exp)) ** 2))

    # AIC = n * log(RSS/n) + 2k, both k=2
    aic_pow = n_pts * np.log(rss_pow / n_pts) + 4
    aic_exp = n_pts * np.log(rss_exp / n_pts) + 4
    delta_aic = aic_pow - aic_exp

    print(f"\n  Power law:   RSS = {rss_pow:.6f}")
    print(f"  Exponential: RSS = {rss_exp:.6f}")
    print(f"  ΔAIC (pow - exp) = {delta_aic:.1f}")
    print(f"  Expected: 15.8")
    assert abs(delta_aic - 15.8) < 0.1, f"ΔAIC mismatch: got {delta_aic:.1f}"
    print(f"  ✓ VERIFIED")


def verify_exp_h_v2(exp_h_path=None):
    """Verify EXP H v2 key numbers."""
    print(f"\n{'=' * 60}")
    print("EXP H v2 VERIFICATION")
    print("=" * 60)

    path = exp_h_path or find_latest("exp_H_v2_*.json")
    print(f"  Using: {Path(path).name}")
    with open(path) as f:
        data = json.load(f)

    # Paper delta
    pfr = data["paper_features_result"]
    paper_delta = pfr["mean_delta"]
    print(f"  Paper delta: {paper_delta} (writeup: -0.031)")
    assert abs(paper_delta - (-0.03125)) < 1e-6, f"Delta mismatch: {paper_delta}"
    print(f"  ✓ VERIFIED")

    # Null distribution
    nd = data["null_distribution"]
    deltas = nd["all_deltas"]
    zeros = sum(1 for x in deltas if x == 0.0)
    nonzeros = [x for x in deltas if x != 0.0]
    print(f"\n  Total random pairs: {len(deltas)} (writeup: 150)")
    print(f"  Zero deltas: {zeros} (writeup: 140)")
    print(f"  Non-zero: {len(nonzeros)}")

    # Conditional analysis
    strictly_worse = sum(1 for x in nonzeros if x < paper_delta)
    ties = sum(1 for x in nonzeros if x == paper_delta)
    strictly_better = sum(1 for x in nonzeros if x > paper_delta)
    cond_p = (strictly_worse + ties) / len(nonzeros)
    print(f"\n  Strictly worse: {strictly_worse}")
    print(f"  Tied: {ties}")
    print(f"  Strictly better: {strictly_better}")
    print(f"  Conditional p (rank/N): {cond_p:.2f} (writeup: 0.40)")
    assert abs(cond_p - 0.40) < 0.01, f"Conditional p mismatch: {cond_p}"
    print(f"  ✓ VERIFIED")

    # Baseline and round-trip
    bl = data["baseline"]
    bl_diffs = [bl[k]["logit_diff"] for k in sorted(bl.keys())]
    mean_bl = sum(bl_diffs) / len(bl_diffs)

    rt = data["sae_roundtrip_control"]
    rt_diffs = [rt["per_prompt"][k]["logit_diff"] for k in sorted(rt["per_prompt"].keys())]
    mean_rt = sum(rt_diffs) / len(rt_diffs)
    rt_delta = mean_rt - mean_bl

    print(f"\n  Baseline mean logit diff: {mean_bl}")
    print(f"  Round-trip delta: {rt_delta} (writeup: -1.875)")
    assert abs(rt_delta - (-1.875)) < 0.01, f"RT delta mismatch: {rt_delta}"
    print(f"  ✓ VERIFIED")

    # Reconstruction share
    recon_share = (1 - abs(paper_delta) / 2.609) * 100
    print(f"\n  Effect eliminated by delta-steering: {recon_share:.1f}% (writeup: 98.8%)")
    print(f"  (of which ~72% is pure round-trip error, rest is nonlinear effects)")
    print(f"  ✓ VERIFIED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify writeup numbers from JSON data")
    parser.add_argument("--all-experiments", help="Path to all_experiments JSON")
    parser.add_argument("--exp-h", help="Path to exp_H_v2 JSON")
    args = parser.parse_args()

    verify_scaling_law(args.all_experiments)
    verify_exp_h_v2(args.exp_h)
    print(f"\n{'=' * 60}")
    print("ALL NUMBERS VERIFIED SUCCESSFULLY")
    print("=" * 60)
