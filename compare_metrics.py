#!/usr/bin/env python3
"""
compare_metrics.py — Domain-wise and leave-one-out comparison of baseline vs ours.

Usage:
    python compare_metrics.py
"""

import json
from pathlib import Path

BASELINE_JSON = Path("F:/Rabia-Salman/EEdit-0/EEDit_baseline/composition_metrics.json")
OURS_JSON     = Path("F:/Rabia-Salman/EEdit-0/EEDit_baseline/ours_composition_metrics.json")

METRICS = ["mse", "psnr", "ssim", "lpips", "clip"]
# For MSE/LPIPS lower is better; for PSNR/SSIM/CLIP higher is better
HIGHER_IS_BETTER = {"mse": False, "psnr": True, "ssim": True, "lpips": False, "clip": True}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def weighted_avg(groups_data, exclude=None):
    """Compute weighted average across groups, optionally excluding one group."""
    totals = {m: 0.0 for m in METRICS}
    n_total = 0
    for group, data in groups_data.items():
        if group == exclude:
            continue
        n = data["count"]
        for m in METRICS:
            totals[m] += data["average"][m] * n
        n_total += n
    if n_total == 0:
        return {m: float("nan") for m in METRICS}, 0
    return {m: totals[m] / n_total for m in METRICS}, n_total


def delta_str(metric, base_val, our_val):
    diff = our_val - base_val
    better = (diff > 0) == HIGHER_IS_BETTER[metric]
    tag = " [+]" if better else " [-]"
    return f"{diff:+.4f}{tag}"


def print_table(title, base_avgs, our_avgs, n_base=None, n_ours=None):
    count_str = ""
    if n_base is not None:
        count_str = f"  (baseline n={n_base}, ours n={n_ours})"
    print(f"\n{'-' * 72}")
    print(f"  {title}{count_str}")
    print(f"{'-' * 72}")
    print(f"  {'Metric':<8}  {'Baseline':>10}  {'Ours':>10}  {'Delta':>16}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*16}")
    for m in METRICS:
        b = base_avgs[m]
        o = our_avgs[m]
        d = delta_str(m, b, o)
        print(f"  {m.upper():<8}  {b:>10.4f}  {o:>10.4f}  {d:>16}")
    print(f"{'-' * 72}")


def main():
    base = load(BASELINE_JSON)
    ours = load(OURS_JSON)

    base_groups = base["per_group"]
    ours_groups = ours["per_group"]

    all_groups = sorted(set(base_groups) | set(ours_groups))

    # -- Domain-wise comparison ----------------------------------------------
    print("\n" + "=" * 72)
    print("  DOMAIN-WISE COMPARISON")
    print("=" * 72)

    for group in all_groups:
        if group not in base_groups:
            print(f"\n  [SKIP] {group}: not in baseline")
            continue
        if group not in ours_groups:
            print(f"\n  [SKIP] {group}: not in ours")
            continue
        b_avg = base_groups[group]["average"]
        o_avg = ours_groups[group]["average"]
        nb = base_groups[group]["count"]
        no = ours_groups[group]["count"]
        print_table(group, b_avg, o_avg, nb, no)

    # -- Overall average -----------------------------------------------------
    print("\n" + "=" * 72)
    print("  OVERALL AVERAGE (all domains)")
    print("=" * 72)

    b_overall, nb_all = weighted_avg(base_groups)
    o_overall, no_all = weighted_avg(ours_groups)
    print_table("ALL DOMAINS", b_overall, o_overall, nb_all, no_all)

    # -- Leave-one-domain-out averages ---------------------------------------
    print("\n" + "=" * 72)
    print("  LEAVE-ONE-DOMAIN-OUT AVERAGES")
    print("=" * 72)

    for exclude in all_groups:
        b_loo, nb_loo = weighted_avg(base_groups, exclude=exclude)
        o_loo, no_loo = weighted_avg(ours_groups, exclude=exclude)
        print_table(f"Excl. {exclude}", b_loo, o_loo, nb_loo, no_loo)


if __name__ == "__main__":
    main()
