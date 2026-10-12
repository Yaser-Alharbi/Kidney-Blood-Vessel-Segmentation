"""Build the static project site into docs/ (served by GitHub Pages).

    python -m demo.build_site                 # full build (needs the v2 splits.json)
    python -m demo.build_site --skip-images   # re-render from docs/build_meta.json

Every number on the page is computed here, never typed into the template:
statistics come from the committed `artifacts_v2/phase3_summary.json` (via
`src.training.paired_stats`, shared with `scripts/report_v2.py`),
`artifacts/inspection_summary.json` and `artifacts/mask_audit.json`. Values
that need local data or extra computation (split sizes, the HEDJitter
settings, the Macenko sign check) are saved to `docs/build_meta.json`, which
`--skip-images` reads back. After rendering, every number in the page text
is checked against the values the template was given.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from demo import data, inference
from src.data.transforms import macenko_sign_check
from src.training.paired_stats import CI, N_BOOT, paired_stats
from src.utils.paths import project_root

DOCS = project_root() / "docs"
SITE_SRC = Path(__file__).resolve().parent / "site"
BASE_ARM, JITTER_ARM = "rgb_aug", "hed_only"
MAC_ARM, FULL_ARM = "macenko_only", "full_stain_aware"
SPLITS = ("test", "val")
ROTATION_DEG = 90  # A.RandomRotate90 in every training pipeline
YEAR = 2026
MINUS = "−"

# arm id -> CSS colour token suffix (--c-base, ...) and key used by site.js
ARM_TOKEN = {BASE_ARM: "base", JITTER_ARM: "hed", MAC_ARM: "mac", FULL_ARM: "full"}

# section, table and figure numbers, rendered through the `label` filter
NUMBERING = {
    "overview": "1", "pipeline": "2", "design": "3", "split": "3.1", "stain": "3.2",
    "arms": "3.3", "training": "3.4", "repro": "3.5", "stats": "3.6",
    "results": "4", "limits": "5", "fig_pipeline": "1", "fig_results": "2",
    "table_results": "1",
}

# numbered in this order; in-text citations use the key
REFERENCES = (
    ("ruifrok", "Ruifrok, A. C. &amp; Johnston, D. A. (2001). Quantification of histochemical "
                "staining by color deconvolution. <i>Analytical and Quantitative Cytology and "
                "Histology</i>, 23(4), 291–299."),
    ("macenko", "Macenko, M. et al. (2009). A method for normalizing histology slides for "
                "quantitative analysis. <i>Proc. IEEE ISBI</i>, 1107–1110."),
    ("tellez", "Tellez, D. et al. (2019). Quantifying the effects of data augmentation and "
               "stain color normalization in convolutional neural networks for computational "
               "pathology. <i>Medical Image Analysis</i>, 58, 101544."),
    ("hubmap", "HuBMAP. <i>Hacking the Human Vasculature</i>. Kaggle competition, 2023."),
)


# ------------------------------------------------------------------ formatting


class Numbers:
    """Jinja filters that format numbers and remember every string they emit."""

    def __init__(self) -> None:
        self.emitted: Set[str] = set()

    def _keep(self, text: str) -> str:
        text = text.replace("-", MINUS)  # true minus sign for negatives
        self.emitted.add(text)
        return text

    def f3(self, x: float) -> str:
        """Three decimals (Dice)."""
        return self._keep(f"{x:.3f}")

    def f2(self, x: float) -> str:
        """Two decimals (effect sizes)."""
        return self._keep(f"{x:.2f}")

    def signed(self, x: float) -> str:
        """Three decimals with a sign (differences)."""
        return self._keep(f"{x:+.3f}")

    def pval(self, p: float) -> str:
        """p-value: 2 decimals from 0.1, 3 from 0.001, 2 sig. figs below, '< 10^-4' under 1e-4."""
        if p < 1e-4:
            return Markup(self._keep(f"&lt; 10<sup>{MINUS}4</sup>"))
        if p >= 0.1:
            return self._keep(f"{p:.2f}")
        if p >= 1e-3:
            return self._keep(f"{p:.3f}")
        return self._keep(f"{p:.{1 - math.floor(math.log10(p))}f}")

    def pct(self, x: float, digits: int = 1) -> str:
        """Fraction as a percentage."""
        return self._keep(f"{x * 100:.{digits}f}%")

    def n(self, x: float) -> str:
        """Integer with thousands separators."""
        return self._keep(f"{int(round(x)):,}")

    def num(self, x: float) -> str:
        """Plain number, shortest form (config values such as 0.05)."""
        return self._keep(f"{x:g}")

    def pow10(self, x: float) -> str:
        """An exact power of ten as 10<sup>k</sup> (learning rate, weight decay)."""
        k = round(math.log10(x))
        if not math.isclose(x, 10.0 ** k):
            raise ValueError(f"{x} is not a power of ten")
        return Markup(self._keep(f"10<sup>{k}</sup>"))

    def label(self, x: str) -> str:
        """Section, table, figure or reference number, or a file path, from this module."""
        self.emitted.add(str(x))
        return str(x)

    def filters(self) -> Dict[str, Callable]:
        """Filter table for Jinja."""
        names = ("f3", "f2", "signed", "pval", "pct", "n", "num", "pow10", "label")
        return {k: getattr(self, k) for k in names}


class _TextOnly(HTMLParser):
    """Collect visible text, skipping <script> and <style>.

    <template> content is collected like any other text, so the numbers in
    the pipeline panel templates are checked too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):  # pylint: disable=redefined-outer-name
        if not self._skip:
            self.parts.append(data)


_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?")


def unexplained_numbers(html: str, emitted: Set[str]) -> List[str]:
    """Numbers in the page text that no filter produced (should be empty)."""
    parser = _TextOnly()
    parser.feed(html)
    text = " ".join(parser.parts)
    allowed = {"34"}  # "ResNet-34" is an architecture name
    for _, ref in REFERENCES:  # bibliographic years, volumes and pages
        allowed.update(_NUMBER.findall(ref))
    for value in emitted:
        allowed.update(_NUMBER.findall(value))
    return sorted(set(_NUMBER.findall(text)) - allowed)


# ------------------------------------------------------------------ statistics


def _results(summary: Dict, arms: List[str]) -> Dict[str, Dict]:
    """Paired statistics per split, keyed by the short arm keys used on the page."""
    out = {}
    for split in SPLITS:
        res = paired_stats(summary, arms, BASE_ARM, split,
                           interaction=(FULL_ARM, JITTER_ARM, MAC_ARM))
        out[split] = {
            "n_seeds": res["n_seeds"], "n_tiles": res["n_tiles"],
            "arms": {ARM_TOKEN[a]: v for a, v in res["arms"].items()},
            "vs": {ARM_TOKEN[a]: v for a, v in res["vs_base"].items()},
            "interaction": res["interaction"],
        }
    return out


def _best(results: Dict) -> Dict[str, str]:
    """Arm key with the highest value in each mean/difference column of Table 1."""
    best = {}
    for split in SPLITS:
        best[f"{split}_dice"] = max(results[split]["arms"],
                                    key=lambda k, s=split: results[s]["arms"][k]["mean"])
        best[f"{split}_delta"] = max(results[split]["vs"],
                                     key=lambda k, s=split: results[s]["vs"][k]["mean"])
    return best


def _figure_data(results: Dict) -> Dict[str, Any]:
    """Per-seed run means and difference CIs for Figure 2 (drawn by site.js)."""
    return {
        "runs": {s: {k: [round(v, 5) for v in a["values"]]
                     for k, a in results[s]["arms"].items()} for s in SPLITS},
        "ci": {s: {k: [round(c["lo"], 5), round(c["hi"], 5)]
                   for k, c in results[s]["vs"].items()} for s in SPLITS},
    }


# ---------------------------------------------------------------------- context


def build_context(summary: Dict, arms: List[str], meta: Dict) -> Dict[str, Any]:
    """Everything the template needs, as raw numbers and plain strings."""
    cfg = data.load_demo_config()
    snap = summary["config_snapshot"]
    with open(project_root() / "artifacts" / "inspection_summary.json", encoding="utf-8") as f:
        inspection = json.load(f)
    with open(project_root() / "artifacts" / "mask_audit.json", encoding="utf-8") as f:
        coverage = json.load(f)["coverage"]
    results = _results(summary, arms)
    summary_path = (cfg.paths.artifacts / "phase3_summary.json").relative_to(project_root())
    return {
        "urls": {"repo": data.REPO_URL, "lab": data.LAB_URL, "weights": data.WEIGHTS_URL},
        "year": YEAR, "sec": NUMBERING, "rot_deg": ROTATION_DEG,
        "refs": [{"key": k, "no": str(i), "html": Markup(text)}
                 for i, (k, text) in enumerate(REFERENCES, start=1)],
        "ref_no": {k: str(i) for i, (k, _) in enumerate(REFERENCES, start=1)},
        "paths": {"raw": "data/raw/", "splits": "src/data/splits.py",
                  "train": "src/training/train.py", "summary": summary_path.as_posix()},
        "arm_id": {ARM_TOKEN[a]: a for a in arms},
        "n_seeds": len(summary["seeds"]), "n_boot": N_BOOT, "ci_level": CI,
        "cfg": {k: snap[k] for k in ("epochs", "early_stop_patience", "lr", "weight_decay",
                                     "image_size")},
        "n_vessels": inspection["annotation_type_counts"]["blood_vessel"],
        "n_annotated": inspection["n_polygon_lines"], "coverage": coverage["mean"],
        "splits": meta["splits"], "hed": meta["hed"], "sign": meta["macenko_sign_check"],
        "r": results, "best": _best(results), "fig": _figure_data(results),
    }


# ---------------------------------------------------------------------- meta pass


def meta_pass() -> Dict[str, Any]:
    """Values that need local data or extra computation (saved as build meta)."""
    cfg = data.load_demo_config()
    with open(cfg.paths.splits, encoding="utf-8") as f:
        splits = json.load(f)
    hed = next(t for t in inference.EVAL_BUILDERS[JITTER_ARM](train=True).compose.transforms
               if type(t).__name__ == "HEDJitter")
    return {
        "splits": {k: (len(v) if isinstance(v, list) else v) for k, v in splits.items()
                   if k in ("train", "val", "test", "train_wsis", "val_wsis", "test_wsis")},
        "hed": {"sigma": hed.sigma, "p": hed.p},
        "macenko_sign_check": macenko_sign_check(),
    }


# ---------------------------------------------------------------------- render


def _to_builtin(obj: Any) -> Any:
    """numpy scalars -> Python numbers for JSON."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def render(context: Dict[str, Any]) -> List[str]:
    """Render docs/index.html, copy static files; return unexplained numbers."""
    numbers = Numbers()
    env = Environment(loader=FileSystemLoader(SITE_SRC), autoescape=select_autoescape(),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(numbers.filters())
    html = env.get_template("index.html.j2").render(**context)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    for name in ("style.css", "site.js"):
        shutil.copy2(SITE_SRC / name, DOCS / name)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    return unexplained_numbers(html, numbers.emitted)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-images", action="store_true",
                        help="re-render HTML from docs/build_meta.json (no local data needed)")
    args = parser.parse_args()
    cfg = data.load_demo_config()
    arms = data.arms_from_config(cfg)
    missing = set(arms) ^ set(ARM_TOKEN)
    if missing:
        raise SystemExit(f"[build_site] arms without a page colour token: {sorted(missing)}")
    summary = data.load_summary(cfg)
    meta_path = DOCS / "build_meta.json"
    if args.skip_images:
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} is missing; run a full build first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = _to_builtin(meta_pass())
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[build_site] Macenko sign check: {meta['macenko_sign_check']}")
    stray = render(_to_builtin(build_context(summary, arms, meta)))
    print(f"[build_site] wrote {DOCS / 'index.html'}")
    if stray:
        raise SystemExit(f"[build_site] numbers in the page with no source: {stray}")
    print("[build_site] number check passed: every number in the page text has a source")


if __name__ == "__main__":
    main()
