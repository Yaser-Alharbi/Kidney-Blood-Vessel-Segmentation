"""Augmentation pipelines (RGB / HED-only / Macenko-only / HED+Macenko).

Each builder returns an Albumentations-shaped callable:
    transform(image=uint8_hwc, mask=uint8_hw)
        -> {"image": float32 [3,H,W], "mask": int64 [H,W], ...}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# Ruifrok & Johnston (2001) RGB<-HED. DAB row is unused for PAS but
# keeps the matrix invertible.
_RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)

# PAS reference: stain matrix + 99th-pct concentrations, per-entry median
# over the v2 train-WSI tiles (`python -m scripts.fit_macenko_ref`).
# Replaces the H&E reference from Macenko (2009) the v2 models used.
_MACENKO_REF_STAIN = np.array(
    [
        [0.5330, 0.2392],
        [0.7490, 0.9477],
        [0.3936, 0.2114],
    ],
    dtype=np.float64,
)
_MACENKO_REF_MAX_C = np.array([0.6946, 0.6840], dtype=np.float64)
_MACENKO_IO = 240.0  # transmitted-light baseline


def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """OD = -log10((I + 1) / Io)."""
    rgb = rgb.astype(np.float64) + 1.0
    return -np.log10(np.maximum(rgb / _MACENKO_IO, 1e-6))


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = (10.0 ** (-od)) * _MACENKO_IO
    return np.clip(rgb, 0.0, 255.0)


class HEDJitter(ImageOnlyTransform):
    """Multiplicative + additive jitter in HED stain space (image only)."""

    def __init__(self, sigma: float = 0.05, p: float = 0.5):
        super().__init__(p=p)
        self.sigma = float(sigma)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.ndim != 3 or img.shape[-1] != 3:
            return img
        h, w, _ = img.shape
        flat = img.reshape(-1, 3)
        od = _rgb_to_od(flat)
        hed = od @ _HED_FROM_RGB
        alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=(1, 3))
        beta = np.random.uniform(-self.sigma, self.sigma, size=(1, 3))
        hed = hed * alpha + beta
        od_new = hed @ _RGB_FROM_HED
        rgb = _od_to_rgb(od_new).reshape(h, w, 3)
        return rgb.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("sigma",)


def hed_shift(
    img: np.ndarray,
    h_scale: float,
    e_scale: float,
    h_bias: float = 0.0,
    e_bias: float = 0.0,
) -> np.ndarray:
    """Deterministic HED shift for the demo's stain stress test.

    Mirrors the maths of `HEDJitter.apply` (hed * alpha + beta) with fixed
    alpha/beta instead of random draws. The D channel is left unchanged,
    whereas training-time `HEDJitter` jitters D too. Keep the two in sync.
    """
    h, w, _ = img.shape
    hed = _rgb_to_od(img.reshape(-1, 3)) @ _HED_FROM_RGB
    hed = hed * np.array([[h_scale, e_scale, 1.0]]) + np.array([[h_bias, e_bias, 0.0]])
    rgb = _od_to_rgb(hed @ _RGB_FROM_HED).reshape(h, w, 3)
    return rgb.astype(np.uint8)


def _macenko_stain_matrix(
    od: np.ndarray, beta: float, alpha_pct: float, fix_sign: bool = False
) -> Optional[np.ndarray]:
    """Estimate a tile's 3x2 stain matrix from its optical densities [N, 3].

    Pixels with no channel above `beta` are dropped as background; the
    stain vectors are the `alpha_pct` angular extremes of the foreground in
    the plane of the top two right-singular vectors. Returns None when the
    tile has too little tissue or the SVD fails.

    Singular vectors are only defined up to sign, so without `fix_sign` the
    angles (and the extremes picked) can change with pixel order. With
    `fix_sign=True` each vector is multiplied by the sign of its component
    sum first. `MacenkoNormalize` uses `fix_sign=True`; the v2 models
    were trained with `fix_sign=False`.
    """
    fg = od[(od > beta).any(axis=1)]
    if fg.shape[0] < 50:
        return None

    try:
        top = np.linalg.svd(
            fg - fg.mean(axis=0, keepdims=True), full_matrices=False
        )[2][:2]
    except np.linalg.LinAlgError:
        return None
    if fix_sign:
        top = top * np.where(top.sum(axis=1, keepdims=True) < 0, -1.0, 1.0)
    plane = top.T  # top-2 singular vectors as 3x2 basis

    proj = fg @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha_pct, 100.0 - alpha_pct])
    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])

    # H-like (more red-absorbing) stain in column 0
    if v_lo[0] > v_hi[0]:
        stain = np.column_stack([v_lo, v_hi])
    else:
        stain = np.column_stack([v_hi, v_lo])
    return stain / (np.linalg.norm(stain, axis=0, keepdims=True) + 1e-8)


def _synthetic_two_stain_tile(rng: np.random.Generator, size: int) -> np.ndarray:
    """RGB uint8 tile from gamma-distributed H and E concentrations (no DAB)."""
    conc_h = np.clip(rng.gamma(2, 0.25, (size, size)), 0, 2)
    conc_e = np.clip(rng.gamma(2, 0.30, (size, size)), 0, 2)
    od = np.stack([conc_h, conc_e, np.zeros_like(conc_h)], axis=-1) @ _RGB_FROM_HED
    return np.clip(10.0 ** (-od) * _MACENKO_IO, 0, 255).astype(np.uint8)


def _flips_and_rotations(rgb: np.ndarray) -> list:
    """The seven non-identity flips/90-degree rotations of a tile."""
    up = np.flipud(rgb)
    return [up, np.fliplr(rgb), np.rot90(rgb, 1), np.rot90(rgb, 2), np.rot90(rgb, 3),
            np.rot90(up, 1), np.rot90(up, 3)]


def macenko_sign_check(
    n_tiles: int = 20, size: int = 256, tol: float = 1e-3, seed: int = 1
) -> Dict[str, int]:
    """Count flip/rotation variants whose estimated stain matrix differs.

    Synthetic two-stain tiles (gamma-distributed H and E concentrations
    mixed with the Ruifrok matrix) are flipped and rotated seven ways.
    Those moves only reorder pixels, so a stable estimator returns the same
    stain matrix for every variant. A variant is inconsistent when any
    entry differs from the original's by more than `tol`. Counted with and
    without the sign fix in `_macenko_stain_matrix`, using the
    `MacenkoNormalize` defaults (beta 0.15, alpha_pct 1).
    """
    def estimate(rgb: np.ndarray, fix: bool) -> np.ndarray:
        od = _rgb_to_od(np.ascontiguousarray(rgb).reshape(-1, 3))
        return _macenko_stain_matrix(od, 0.15, 1.0, fix_sign=fix)

    rng = np.random.default_rng(seed)
    counts = {"n_variants": 0, "inconsistent": 0, "inconsistent_fixed": 0}
    for _ in range(n_tiles):
        rgb = _synthetic_two_stain_tile(rng, size)
        variants = _flips_and_rotations(rgb)
        counts["n_variants"] += len(variants)
        for key, fix in (("inconsistent", False), ("inconsistent_fixed", True)):
            ref = estimate(rgb, fix)
            counts[key] += sum(int(np.abs(estimate(v, fix) - ref).max() > tol)
                               for v in variants)
    return counts


class MacenkoNormalize(ImageOnlyTransform):
    """Macenko stain normalization to a fixed reference (image only).

    Falls back to the input image on degenerate tiles (SVD/lstsq fail).
    """

    def __init__(self, beta: float = 0.15, alpha_pct: float = 1.0, p: float = 1.0):
        super().__init__(p=p)
        self.beta = float(beta)
        self.alpha_pct = float(alpha_pct)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.ndim != 3 or img.shape[-1] != 3:
            return img
        h, w, _ = img.shape
        flat = img.reshape(-1, 3)
        od = _rgb_to_od(flat)

        stain = _macenko_stain_matrix(od, self.beta, self.alpha_pct, fix_sign=True)
        if stain is None:
            return img

        try:
            conc, *_ = np.linalg.lstsq(stain, od.T, rcond=None)
        except np.linalg.LinAlgError:
            return img

        max_c = np.maximum(np.percentile(conc, 99.0, axis=1), 1e-6)
        conc = conc * (_MACENKO_REF_MAX_C[:, None] / max_c[:, None])
        od_norm = (_MACENKO_REF_STAIN @ conc).T
        return _od_to_rgb(od_norm).reshape(h, w, 3).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("beta", "alpha_pct")


class _MaskedCompose:
    """Wrap a Compose; coerce mask to int64 (0, 1)."""

    def __init__(self, compose: A.Compose) -> None:
        self.compose = compose

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        out = self.compose(image=image, mask=mask)
        m = out["mask"]
        if isinstance(m, torch.Tensor):
            out["mask"] = (m > 0).long()
        else:
            out["mask"] = (np.asarray(m) > 0).astype(np.int64)
        return out


def get_rgb_aug_transforms(image_size: int = 512, train: bool = True) -> _MaskedCompose:
    """RGB control: flips + rotate90 + brightness/contrast."""
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_hed_only_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """Ablation: HED jitter on train, no Macenko on eval."""
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            HEDJitter(sigma=0.05, p=0.8),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_macenko_only_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """Isolates Macenko: stain normalization on train + eval, no HED jitter.

    Reuses the same fixed reference matrix as full_stain_aware via
    `MacenkoNormalize` (defined above with `_MACENKO_REF_STAIN` /
    `_MACENKO_REF_MAX_C`).
    """
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_full_stain_aware_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """Combined arm: Macenko then HED jitter on train, Macenko on eval.

    Jitter runs after normalisation so it perturbs around the reference
    stain (the other order would let Macenko undo it), and train and eval
    inputs share the normalised distribution.
    """
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            MacenkoNormalize(p=1.0),
            HEDJitter(sigma=0.05, p=0.8),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


# back-compat aliases (older code paths used the longer names)
get_stain_aware_hed_only_transforms = get_hed_only_transforms
get_stain_aware_transforms = get_full_stain_aware_transforms
