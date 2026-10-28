from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _resolve_weights(weights: Mapping[str, float]) -> dict[str, float]:
    return {
        "wM": float(weights.get("wM", weights.get("M", 0.0))),
        "wG": float(weights.get("wG", weights.get("G", 0.0))),
        "wR": float(weights.get("wR", weights.get("R", 0.0))),
        "wT": float(weights.get("wT", weights.get("T", 0.0))),
    }


def compute_log_risk(M, U, T, G, weights, eps):
    w = _resolve_weights(weights)
    Mc = np.maximum(np.asarray(M, dtype=float), eps)
    Uc = np.maximum(np.asarray(U, dtype=float), eps)
    Tc = np.maximum(np.asarray(T, dtype=float), eps)
    Gc = np.maximum(np.asarray(G, dtype=float), eps)
    return (
        w["wM"] * np.log(Mc)
        + w["wG"] * np.log(Gc)
        + w["wR"] * np.log(Uc)
        + w["wT"] * np.log(Tc)
    )


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def scale_scores(log_risk, method):
    x = np.asarray(log_risk, dtype=float)
    if isinstance(method, Mapping):
        name = str(method.get("name", "mad_tanh"))
        p_lo = float(method.get("p_lo", 5.0))
        p_hi = float(method.get("p_hi", 95.0))
    else:
        name = str(method)
        p_lo, p_hi = 5.0, 95.0

    if name == "mad_tanh":
        med = np.median(x)
        scale = _mad(x) or 1e-12
        return 50.0 * (np.tanh((x - med) / scale) + 1.0)
    if name == "z_tanh":
        mu = np.mean(x)
        sd = np.std(x) or 1e-12
        return 50.0 * (np.tanh((x - mu) / sd) + 1.0)
    if name == "minmax":
        lo, hi = np.min(x), np.max(x)
        if hi - lo == 0:
            return np.full_like(x, 50.0)
        return 100.0 * (x - lo) / (hi - lo)
    if name == "pctclip":
        lo = np.percentile(x, p_lo)
        hi = np.percentile(x, p_hi)
        if hi - lo == 0:
            return np.full_like(x, 50.0)
        return 100.0 * (np.clip(x, lo, hi) - lo) / (hi - lo)

    raise ValueError(f"Unknown scaling method: {name}")


def classify(score, bins: Iterable[float]):
    score_arr = np.asarray(score, dtype=float)
    cuts = list(bins)
    if len(cuts) != 6:
        raise ValueError("bins must include 6 boundaries for classes A-E")
    labels = ["A", "B", "C", "D", "E"]
    extended = [-np.inf, *cuts[1:-1], np.inf]
    return pd.cut(score_arr, bins=extended, labels=labels, right=False)


def compute_rank(score):
    return pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
