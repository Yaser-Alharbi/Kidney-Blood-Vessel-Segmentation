"""Fit the PAS Macenko reference from the v2 train split.

The original reference in `src/data/transforms.py` is the H&E one from
Macenko (2009); HuBMAP kidney tiles are PAS-stained. This estimates a
stain matrix and 99th-percentile concentrations per train tile (sign
fixed), takes the per-entry median across tiles and prints constants to
paste over `_MACENKO_REF_STAIN` / `_MACENKO_REF_MAX_C`. Only the train
WSI is used, so val and test never inform the reference.

    python -m scripts.fit_macenko_ref
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from PIL import Image

from src.data.transforms import _macenko_stain_matrix, _rgb_to_od
from src.utils.paths import load_config, project_root

# MacenkoNormalize defaults
_BETA = 0.15
_ALPHA_PCT = 1.0


def fit() -> Dict[str, object]:
    """Median stain matrix and max concentrations over the train tiles."""
    cfg = load_config(project_root() / "configs" / "rerun_v2.yaml")
    with open(cfg.paths.splits, "r") as f:
        tile_ids: List[str] = json.load(f)["train"]

    stains, max_cs = [], []
    for tile_id in tile_ids:
        img = np.array(Image.open(cfg.paths.raw / "train" / f"{tile_id}.tif").convert("RGB"))
        od = _rgb_to_od(img.reshape(-1, 3))
        stain = _macenko_stain_matrix(od, _BETA, _ALPHA_PCT, fix_sign=True)
        if stain is None:
            continue
        conc, *_ = np.linalg.lstsq(stain, od.T, rcond=None)
        stains.append(stain)
        max_cs.append(np.percentile(conc, 99.0, axis=1))

    stain = np.median(np.stack(stains), axis=0)
    stain = stain / np.linalg.norm(stain, axis=0, keepdims=True)
    return {
        "n_tiles": len(tile_ids),
        "n_used": len(stains),
        "stain": stain,
        "max_c": np.median(np.stack(max_cs), axis=0),
    }


def main() -> None:
    """Print the fitted reference as paste-ready constants."""
    ref = fit()
    print(f"[fit_macenko_ref] tiles used : {ref['n_used']} / {ref['n_tiles']}")
    print("_MACENKO_REF_STAIN = np.array(\n    [")
    for row in ref["stain"]:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}],")
    print("    ],\n    dtype=np.float64,\n)")
    max_c = ref["max_c"]
    print(f"_MACENKO_REF_MAX_C = np.array([{max_c[0]:.4f}, {max_c[1]:.4f}], "
          "dtype=np.float64)")


if __name__ == "__main__":
    main()
