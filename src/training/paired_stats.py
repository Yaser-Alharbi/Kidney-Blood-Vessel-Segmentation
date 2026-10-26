"""Seed-level paired comparisons of arms against a baseline arm.

Used by `scripts/report_v2.py` for the markdown report. The unit of
analysis is a run: its mean per-tile Dice on one split. Runs that share a seed share
their initialisation, so every arm is compared with the baseline seed by
seed:

    paired difference  arm - baseline, per seed
    95% CI             percentile bootstrap of the mean difference
    p                  two-sided Wilcoxon signed-rank on the paired runs
    p_holm             Holm-adjusted across the arms compared
    d_z                mean difference / s.d. of the differences
    interaction        full - a - b + baseline, per seed (additivity check)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

from src.training.evaluate import bootstrap_dice

N_BOOT = 10_000
CI = 0.95
BOOT_SEED = 42


def _run_tag(arm: str, seed: int) -> str:
    return f"unet_resnet34_{arm}_seed{seed}"


def per_run_means(
    summary: Dict[str, Any], arms: Sequence[str], split: str
) -> Tuple[List[int], Dict[str, np.ndarray], int]:
    """Mean per-tile Dice of every run, as {arm: array over seeds}.

    Returns the seeds (in summary order), the per-arm arrays and the shared
    tile count. Raises if a run is missing or runs disagree on the number of
    tiles, since paired comparisons need the same seeds and the same tiles.
    """
    seeds = [int(s) for s in summary["seeds"]]
    runs = summary["runs"]
    means: Dict[str, np.ndarray] = {}
    tile_counts = set()
    for arm in arms:
        vals = []
        for seed in seeds:
            tiles = runs[_run_tag(arm, seed)][f"{split}_per_tile_dice"]
            tile_counts.add(len(tiles))
            vals.append(float(np.mean(tiles)))
        means[arm] = np.array(vals, dtype=np.float64)
    if len(tile_counts) != 1:
        raise RuntimeError(f"runs disagree on {split}-set size: {sorted(tile_counts)}")
    return seeds, means, tile_counts.pop()


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjustment (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    k = len(order)
    out: Dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (k - i) * pvalues[key]))
        out[key] = running
    return out


def _mean_ci(values: np.ndarray) -> Dict[str, float]:
    ci = bootstrap_dice(values, n_boot=N_BOOT, ci=CI, seed=BOOT_SEED)
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi}


def _compare(arm: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    """Paired difference, its CI, d_z and the unadjusted two-sided Wilcoxon."""
    diff = arm - base
    res = wilcoxon(arm, base, alternative="two-sided")
    return {
        "diffs": diff.tolist(), **_mean_ci(diff),
        "improved": int((diff > 0).sum()), "n": int(diff.size),
        "dz": float(diff.mean() / diff.std(ddof=1)),
        "statistic": float(res.statistic), "p": float(res.pvalue),
    }


def paired_stats(
    summary: Dict[str, Any],
    arms: Sequence[str],
    base: str,
    split: str,
    interaction: Optional[Tuple[str, str, str]] = None,
) -> Dict[str, Any]:
    """Per-arm summaries and paired comparisons with `base` on one split.

    `interaction` names (combined, a, b); when given, the per-seed contrast
    combined - a - b + base and its bootstrap CI are added. Each bootstrap
    uses its own fixed-seed RNG, so results do not depend on call order.
    """
    seeds, means, n_tiles = per_run_means(summary, arms, split)
    out: Dict[str, Any] = {
        "split": split, "seeds": seeds, "n_seeds": len(seeds), "n_tiles": n_tiles,
        "n_boot": N_BOOT, "ci": CI, "base": base,
        "arms": {a: {"values": means[a].tolist(), "mean": float(means[a].mean()),
                     "std": float(means[a].std(ddof=1))} for a in arms},
        "vs_base": {a: _compare(means[a], means[base]) for a in arms if a != base},
    }
    for arm, p_adj in holm({k: v["p"] for k, v in out["vs_base"].items()}).items():
        out["vs_base"][arm]["p_holm"] = p_adj
    if interaction is not None:
        full, a, b = (means[k] for k in interaction)
        contrast = full - a - b + means[base]
        out["interaction"] = {"values": contrast.tolist(), **_mean_ci(contrast)}
    return out
