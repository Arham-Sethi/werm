from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import cv2


@dataclass
class TillageConfig:
    resize_to: int = 224
    roi_frac: float = 1.0
    patch_size: int = 56
    glcm_levels: int = 32
    use_log_glcm: bool = True
    outlier_percent: float = 15.0  # remove top 15% highest patches


def _glcm_contrast(gray_uint8: np.ndarray, levels: int) -> float:
    g = (gray_uint8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    a = g[:, :-1]
    b = g[:, 1:]

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.ravel(), b.ravel()), 1.0)

    s = float(M.sum())
    if s <= 0:
        return 0.0
    M /= s

    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    return float(((i - j) ** 2 * M).sum())


def compute_tillage_from_rgb(
    rgb_uint8: np.ndarray,
    normalizer=None,
    cfg: Optional[TillageConfig] = None,
    residue_pred: Optional[float] = None,
    alpha: float = 1.5
) -> Dict:

    cfg = cfg or TillageConfig()

    arr = cv2.resize(rgb_uint8, (cfg.resize_to, cfg.resize_to))
    gray_full = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    ps = cfg.patch_size
    H, W = gray_full.shape

    glcm_vals = []

    for y in range(0, H - ps + 1, ps):
        for x in range(0, W - ps + 1, ps):
            patch = gray_full[y:y+ps, x:x+ps]
            g = _glcm_contrast(patch, cfg.glcm_levels)
            if cfg.use_log_glcm:
                g = np.log1p(g)
            glcm_vals.append(g)

    glcm_vals = np.array(glcm_vals)

    if len(glcm_vals) == 0:
        return {"tillage_raw": 0.0, "tillage_eff": 0.0}

    # Remove extreme outliers (likely straw clusters)
    cutoff = np.percentile(glcm_vals, 100 - cfg.outlier_percent)
    filtered = glcm_vals[glcm_vals <= cutoff]

    if len(filtered) == 0:
        filtered = glcm_vals

    mean_glcm = float(np.mean(filtered))

    till_raw = normalizer.transform_value(mean_glcm) if normalizer else mean_glcm

    if residue_pred is not None:
        R = float(np.clip(residue_pred, 0.0, 1.0))
        till_eff = till_raw * ((1.0 - R) ** alpha)
    else:
        till_eff = till_raw

    confidence = len(filtered) / len(glcm_vals)

    return {
        "tillage_raw": till_raw,
        "tillage_eff": till_eff,
        "tillage_conf": confidence,
        "patches_total": len(glcm_vals),
        "patches_used": len(filtered)
    }


class TillageNormalizer:
    def __init__(self, lo_p=2.0, hi_p=98.0):
        self.lo_p = lo_p
        self.hi_p = hi_p
        self.lo = None
        self.hi = None

    def fit_from_image_paths(self, paths: List[str], cfg: Optional[TillageConfig] = None):
        cfg = cfg or TillageConfig()
        vals = []

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = compute_tillage_from_rgb(rgb, normalizer=None, cfg=cfg)
            vals.append(out["tillage_raw"])

        x = np.array(vals)
        self.lo = np.percentile(x, self.lo_p)
        self.hi = np.percentile(x, self.hi_p)
        if self.hi - self.lo < 1e-6:
            self.hi = self.lo + 1.0

    def transform_value(self, val):
        v = (val - self.lo) / (self.hi - self.lo)
        return float(np.clip(v, 0, 1))