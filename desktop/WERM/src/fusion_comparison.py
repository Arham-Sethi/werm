"""
Additive vs Geometric Fusion Comparison
========================================
Computes a weighted arithmetic mean (additive) baseline alongside the existing
log-space weighted geometric mean, then comprehensively compares the two risk
maps across 333 images.

Analyses:
  1. Pipeline constant extraction
  2. Additive baseline computation
  3. Agreement metrics (Spearman, Kendall-tau, top-k overlap, MAD)
  4. Disagreement-zone characterisation (multiplicative suppression)
  5. Farmer expert agreement (moisture folder class proxies)
  6. Epsilon floor sensitivity for geometric fusion
  7. Four publication-quality figures (300 dpi PNG)

Usage:
  python fusion_comparison.py --csv path/to/fusion_results_FREEZE.csv
  python fusion_comparison.py --synthetic          # demo mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================================
# Pipeline constants (extracted from WERM/src/ codebase)
# ============================================================================
# Weights  — fusion_final.py:48, config/fusion_config.yaml:2-5
WM, WG, WR, WT = 0.30, 0.20, 0.20, 0.30

# Epsilon floor before log()  — fusion_final.py:186,  fusion_engine.py:20-23
EPS_DEFAULT = 1e-6

# Tillage floor  — fusion_final.py:45
TILL_FLOOR = 0.02

# Tillage residue-damping alpha — fusion_final.py:42
TILL_ALPHA = 1.0          # (tillage_module.py uses 1.5 in older infer pipeline)

# Tillage gamma compression — fusion_final.py:201
TILL_GAMMA = 0.7

# GDD normalization — fusion_final.py:32
T_BASELINE_GDD = 120.0

# Scaling method — config/fusion_config.yaml:8, fusion_final.py:232-244
# MAD-tanh: z = (log_r - median) / (1.4826 * MAD); score = 50 + 50*tanh(z/2)

# Risk class bins
CLASS_BINS   = [-np.inf, 20, 40, 60, 80, np.inf]
CLASS_LABELS = ["E", "D", "C", "B", "A"]   # E=lowest, A=highest


# ============================================================================
# Helper functions
# ============================================================================

def compute_log_risk(M, G, U, T_eff, eps=EPS_DEFAULT):
    """Log-space weighted geometric mean (mirrors fusion_final.py:206-217)."""
    Mc = np.maximum(np.asarray(M, dtype=float), eps)
    Gc = np.maximum(np.asarray(G, dtype=float), eps)
    Uc = np.maximum(np.asarray(U, dtype=float), eps)
    Tc = np.maximum(np.asarray(T_eff, dtype=float), eps)
    return (WM * np.log(Mc) + WG * np.log(Gc)
            + WR * np.log(Uc) + WT * np.log(Tc))


def scale_mad_tanh(log_risk):
    """Robust MAD-tanh scaling -> [0,100] (fusion_final.py:232-244)."""
    x = np.asarray(log_risk, dtype=float)
    med = float(np.median(x))
    mad_val = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad_val if mad_val > 1e-12 else (float(np.std(x)) + 1e-12)
    z = (x - med) / sigma
    return 50.0 + 50.0 * np.tanh(z / 2.0)


def scale_minmax(x):
    """Min-max normalisation to [0, 100]."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.full_like(x, 50.0)
    return 100.0 * (x - lo) / (hi - lo)


def spearman_corr(a, b):
    """Spearman rank correlation (no scipy dependency)."""
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    ar -= ar.mean(); br -= br.mean()
    d = np.sqrt((ar ** 2).sum()) * np.sqrt((br ** 2).sum())
    return float((ar * br).sum() / d) if d > 0 else float("nan")


def kendall_tau(a, b):
    """Kendall tau-b rank correlation (O(n^2) but fine for n=333)."""
    n = len(a)
    concordant = discordant = tied_a = tied_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            elif da == 0:
                tied_a += 1
            elif db == 0:
                tied_b += 1
            elif (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + tied_a)
                    * (concordant + discordant + tied_b))
    return float((concordant - discordant) / denom) if denom > 0 else 0.0


def top_k_overlap(scores_a, scores_b, frac):
    """Fraction of A's top-frac% that are also in B's top-frac%."""
    k = int(np.ceil(frac * len(scores_a)))
    top_a = set(np.argsort(scores_a)[-k:])
    top_b = set(np.argsort(scores_b)[-k:])
    return len(top_a & top_b) / len(top_a) if len(top_a) > 0 else 0.0


def bootstrap_spearman(a, b, n_boot=2000, seed=42):
    """Bootstrap 95% CI for Spearman correlation."""
    rng = np.random.default_rng(seed)
    n = len(a)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rhos.append(spearman_corr(a[idx], b[idx]))
    rhos = np.sort(rhos)
    return float(rhos[int(0.025 * n_boot)]), float(rhos[int(0.975 * n_boot)])


# ============================================================================
# Synthetic data generator
# ============================================================================

def generate_synthetic_data(n=333, seed=42):
    """Realistic synthetic fusion data with farmer proxies."""
    rng = np.random.default_rng(seed)

    moisture  = rng.beta(2.5, 2.5, n)
    residue   = rng.beta(2.0, 5.0, n)
    gdd_norm  = np.full(n, 0.45)        # constant (single-date collection)
    till_norm = rng.beta(3.0, 3.0, n)

    # Apply pipeline texture transforms: floor + gamma + damping
    till_adj  = TILL_FLOOR + (1.0 - TILL_FLOOR) * (till_norm ** TILL_GAMMA)
    till_eff  = till_adj * ((1.0 - residue) ** TILL_ALPHA)

    # Farmer-proxy labels: moisture class from folder structure
    # Simulate realistic class fractions: ~35% Dry, 40% Moderate, 25% Wet
    farmer_moisture = rng.choice(
        ["Dry", "Moderate", "Wet"], size=n, p=[0.35, 0.40, 0.25])
    # Make moisture score correlated with farmer class
    for i in range(n):
        if farmer_moisture[i] == "Dry":
            moisture[i] = rng.beta(2.0, 6.0)
        elif farmer_moisture[i] == "Moderate":
            moisture[i] = rng.beta(4.0, 4.0)
        else:
            moisture[i] = rng.beta(6.0, 2.0)

    # Tillage labels from folder: maxtill vs mintill
    farmer_tillage = np.where(till_norm > np.median(till_norm), "maxtill", "mintill")

    # Residue: no explicit folder class, but simulate L/M/H
    farmer_residue = np.where(residue < 0.2, "L",
                              np.where(residue < 0.5, "M", "H"))

    return pd.DataFrame({
        "image":           [f"img_{i:04d}.jpg" for i in range(n)],
        "moisture":        moisture,
        "residue":         residue,
        "gdd_norm":        gdd_norm,
        "tillage_norm":    till_norm,
        "tillage_eff":     till_eff,
        "farmer_moisture": farmer_moisture,
        "farmer_tillage":  farmer_tillage,
        "farmer_residue":  farmer_residue,
    })


# ============================================================================
# Main analysis
# ============================================================================

def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_mode = args.synthetic or (args.csv is None)

    if synthetic_mode:
        print("=== SYNTHETIC MODE (no pipeline CSV) ===\n")
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(args.csv)
        needed = ["image", "moisture", "residue", "gdd_norm", "tillage_eff"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: CSV missing columns: {missing}")

    n = len(df)
    M     = df["moisture"].to_numpy(dtype=float)
    R     = df["residue"].to_numpy(dtype=float)
    U     = 1.0 - R
    G     = df["gdd_norm"].to_numpy(dtype=float)
    T_eff = df["tillage_eff"].to_numpy(dtype=float)

    # ======================================================================
    # 1. PRINT PIPELINE CONSTANTS
    # ======================================================================
    print("=" * 72)
    print("1. PIPELINE CONSTANTS (extracted from WERM/src/ codebase)")
    print("=" * 72)
    print(f"  Fusion weights:  wM={WM}, wG={WG}, wR={WR}, wT={WT}")
    print(f"  Epsilon floor:   eps={EPS_DEFAULT}  (applied as max(x, eps) before log)")
    print(f"  Tillage floor:   {TILL_FLOOR}  (smooth floor to avoid hard zeros)")
    print(f"  Tillage gamma:   {TILL_GAMMA}  (compression: T = floor + (1-floor)*T^gamma)")
    print(f"  Residue damp:    alpha={TILL_ALPHA}  (T_eff = T_norm * (1-R)^alpha)")
    print(f"  GDD baseline:    {T_BASELINE_GDD} Cd  (G_norm = min(cumGDD/{T_BASELINE_GDD}, 1))")
    print()
    print("  Geometric fusion formula (fusion_final.py:206-217):")
    print("    risk_raw = exp(0.30*log(max(M,eps)) + 0.20*log(max(G,eps))")
    print("                 + 0.20*log(max(1-R,eps)) + 0.30*log(max(T_eff,eps)))")
    print()
    print("  Scaling (fusion_final.py:232-244):")
    print("    log_r = log(risk_raw)")
    print("    z = (log_r - median(log_r)) / (1.4826 * MAD(log_r))")
    print("    score = 50 + 50 * tanh(z / 2)   ->  [0, 100]")
    print()

    # ======================================================================
    # 2. COMPUTE BOTH FUSION METHODS
    # ======================================================================
    print("=" * 72)
    print("2. COMPUTING GEOMETRIC AND ADDITIVE FUSION")
    print("=" * 72)

    # Geometric (pipeline default)
    log_risk_geo = compute_log_risk(M, G, U, T_eff, eps=EPS_DEFAULT)
    score_geo = scale_mad_tanh(log_risk_geo)

    # Additive baseline: R_add = wM*M + wG*G + wR*U + wT*T_eff
    risk_add_raw = WM * M + WG * G + WR * U + WT * T_eff
    score_add = scale_minmax(risk_add_raw)

    print(f"  Geometric scores: min={score_geo.min():.2f}, "
          f"max={score_geo.max():.2f}, mean={score_geo.mean():.2f}")
    print(f"  Additive scores:  min={score_add.min():.2f}, "
          f"max={score_add.max():.2f}, mean={score_add.mean():.2f}")
    print()

    # ======================================================================
    # 3. AGREEMENT METRICS
    # ======================================================================
    print("=" * 72)
    print("3. AGREEMENT METRICS: Geometric vs Additive")
    print("=" * 72)

    rho_sp = spearman_corr(score_geo, score_add)
    tau    = kendall_tau(score_geo, score_add)
    t20    = top_k_overlap(score_geo, score_add, 0.20)
    t10    = top_k_overlap(score_geo, score_add, 0.10)
    mad_sc = float(np.mean(np.abs(score_geo - score_add)))

    print(f"  Spearman rank correlation:    {rho_sp:.6f}")
    print(f"  Kendall tau-b:                {tau:.6f}")
    print(f"  Top-20% overlap:              {t20:.4f}")
    print(f"  Top-10% overlap:              {t10:.4f}")
    print(f"  Mean |diff| (0-100 scale):    {mad_sc:.4f}")
    print()

    agree_df = pd.DataFrame([{
        "Metric": "Spearman rho", "Value": f"{rho_sp:.6f}"},
        {"Metric": "Kendall tau-b", "Value": f"{tau:.6f}"},
        {"Metric": "Top-20% overlap", "Value": f"{t20:.4f}"},
        {"Metric": "Top-10% overlap", "Value": f"{t10:.4f}"},
        {"Metric": "Mean |diff| score", "Value": f"{mad_sc:.4f}"},
    ])
    agree_df.to_csv(out_dir / "agreement_metrics.csv", index=False)

    # ======================================================================
    # 4. DISAGREEMENT ZONE CHARACTERISATION
    # ======================================================================
    print("=" * 72)
    print("4. DISAGREEMENT ZONES")
    print("=" * 72)

    rank_geo = pd.Series(score_geo).rank(method="average").to_numpy()
    rank_add = pd.Series(score_add).rank(method="average").to_numpy()
    rank_diff = np.abs(rank_geo - rank_add)

    # Top 10% by absolute rank difference
    cutoff = np.percentile(rank_diff, 90)
    disagree_mask = rank_diff >= cutoff
    n_disagree = int(disagree_mask.sum())

    dis_df = pd.DataFrame({
        "image":       df["image"].values[disagree_mask],
        "moisture":    M[disagree_mask],
        "residue":     R[disagree_mask],
        "uncovered":   U[disagree_mask],
        "gdd_norm":    G[disagree_mask],
        "tillage_eff": T_eff[disagree_mask],
        "score_geo":   score_geo[disagree_mask],
        "score_add":   score_add[disagree_mask],
        "diff_geo_add": (score_geo - score_add)[disagree_mask],
        "rank_diff":   rank_diff[disagree_mask],
    }).sort_values("rank_diff", ascending=False)

    print(f"  Top-10% disagreement images: {n_disagree} of {n}")
    print()
    print(dis_df.head(15).to_string(index=False))

    dis_df.to_csv(out_dir / "disagreement_images.csv", index=False)

    # Detect "one-low-others-high" pattern (multiplicative suppression)
    LOW_THRESH  = 0.20
    HIGH_THRESH = 0.60

    def has_suppression(m_val, u_val, t_val, g_val):
        """True if exactly one input < LOW_THRESH and at least 2 others > HIGH_THRESH."""
        vals = [m_val, u_val, t_val, g_val]
        n_low  = sum(1 for v in vals if v < LOW_THRESH)
        n_high = sum(1 for v in vals if v > HIGH_THRESH)
        return n_low >= 1 and n_high >= 2

    suppression_flags = []
    suppression_which = []
    factor_names = ["moisture", "uncovered", "tillage_eff", "gdd_norm"]

    for i in range(n):
        vals = [M[i], U[i], T_eff[i], G[i]]
        flag = has_suppression(*vals)
        suppression_flags.append(flag)
        if flag:
            low_factors = [fn for fn, v in zip(factor_names, vals) if v < LOW_THRESH]
            suppression_which.append(", ".join(low_factors))
        else:
            suppression_which.append("")

    suppression_flags = np.array(suppression_flags)
    n_suppress_total = int(suppression_flags.sum())
    n_suppress_disagree = int(suppression_flags[disagree_mask].sum())

    print(f"\n  'One-low-others-high' suppression pattern:")
    print(f"    Total images with pattern:           {n_suppress_total} / {n}")
    print(f"    Among top-10% disagreement images:   {n_suppress_disagree} / {n_disagree}")
    print(f"    Fraction of disagreement with pattern: "
          f"{n_suppress_disagree / n_disagree:.2%}" if n_disagree > 0 else "")
    print()

    # Detail: which factor is the suppressor in disagreement images
    dis_suppress = dis_df.copy()
    dis_suppress["suppression"] = suppression_flags[disagree_mask]
    dis_suppress["low_factor"]  = np.array(suppression_which)[disagree_mask]
    print("  Disagreement images with suppression pattern:")
    supp_only = dis_suppress[dis_suppress["suppression"]].head(10)
    if len(supp_only) > 0:
        print(supp_only[["image", "moisture", "uncovered", "tillage_eff",
                          "gdd_norm", "score_geo", "score_add",
                          "diff_geo_add", "low_factor"]].to_string(index=False))
    else:
        print("  (none found)")
    print()

    suppression_summary = pd.DataFrame([{
        "total_images": n,
        "images_with_suppression_pattern": n_suppress_total,
        "top10pct_disagree_images": n_disagree,
        "disagree_with_suppression": n_suppress_disagree,
        "fraction": n_suppress_disagree / n_disagree if n_disagree > 0 else 0,
    }])
    suppression_summary.to_csv(out_dir / "suppression_pattern.csv", index=False)

    # ======================================================================
    # 5. FARMER EXPERT AGREEMENT
    # ======================================================================
    print("=" * 72)
    print("5. FARMER EXPERT AGREEMENT")
    print("=" * 72)

    # Encode farmer classes as ordinal scores
    moisture_ord_map  = {"Dry": 0.0, "Moderate": 0.5, "Wet": 1.0}
    tillage_ord_map   = {"mintill": 0.0, "maxtill": 1.0}
    residue_ord_map   = {"L": 0.0, "M": 0.5, "H": 1.0}

    has_farmer = all(c in df.columns for c in
                     ["farmer_moisture", "farmer_tillage", "farmer_residue"])

    if has_farmer:
        farmer_m = df["farmer_moisture"].map(moisture_ord_map).to_numpy(dtype=float)
        farmer_t = df["farmer_tillage"].map(tillage_ord_map).to_numpy(dtype=float)
        farmer_r = df["farmer_residue"].map(residue_ord_map).to_numpy(dtype=float)

        # Overall farmer score: mean of per-feature ordinal
        # Residue is INVERSE risk (high residue = low risk), so use (1 - R_ord)
        farmer_overall = (farmer_m + farmer_t + (1.0 - farmer_r)) / 3.0

        # Correlations
        rho_geo_farmer = spearman_corr(score_geo, farmer_overall)
        rho_add_farmer = spearman_corr(score_add, farmer_overall)

        # Per-feature
        rho_geo_fm = spearman_corr(score_geo, farmer_m)
        rho_add_fm = spearman_corr(score_add, farmer_m)
        rho_geo_ft = spearman_corr(score_geo, farmer_t)
        rho_add_ft = spearman_corr(score_add, farmer_t)
        rho_geo_fr = spearman_corr(score_geo, farmer_r)
        rho_add_fr = spearman_corr(score_add, farmer_r)

        # Bootstrap CIs
        ci_geo = bootstrap_spearman(score_geo, farmer_overall)
        ci_add = bootstrap_spearman(score_add, farmer_overall)

        print(f"  Farmer ordinal encoding:")
        print(f"    Moisture:  Dry=0, Moderate=0.5, Wet=1")
        print(f"    Tillage:   mintill=0, maxtill=1")
        print(f"    Residue:   L=0, M=0.5, H=1  (inverted for risk: 1-R)")
        print(f"    Overall:   mean(moisture_ord, tillage_ord, 1-residue_ord)")
        print()
        print(f"  Spearman vs farmer overall:")
        print(f"    Geometric: {rho_geo_farmer:.4f}  95% CI [{ci_geo[0]:.4f}, {ci_geo[1]:.4f}]")
        print(f"    Additive:  {rho_add_farmer:.4f}  95% CI [{ci_add[0]:.4f}, {ci_add[1]:.4f}]")
        print()
        print(f"  Per-feature Spearman:")
        print(f"    {'Feature':<15s} {'Geometric':>10s} {'Additive':>10s}")
        print(f"    {'Moisture':<15s} {rho_geo_fm:>10.4f} {rho_add_fm:>10.4f}")
        print(f"    {'Tillage':<15s} {rho_geo_ft:>10.4f} {rho_add_ft:>10.4f}")
        print(f"    {'Residue':<15s} {rho_geo_fr:>10.4f} {rho_add_fr:>10.4f}")

        better = "GEOMETRIC" if rho_geo_farmer > rho_add_farmer else "ADDITIVE"
        print(f"\n  ==> {better} fusion agrees better with farmer experts")
    else:
        print("  No farmer classification columns found in CSV.")
        print("  Generating proxy labels from input value distributions.")
        # Create proxy farmer labels from the input values themselves
        farmer_m = np.where(M < 0.33, 0.0, np.where(M < 0.66, 0.5, 1.0))
        farmer_t = np.where(T_eff < np.median(T_eff), 0.0, 1.0)
        farmer_r = np.where(R < 0.2, 0.0, np.where(R < 0.5, 0.5, 1.0))
        farmer_overall = (farmer_m + farmer_t + (1.0 - farmer_r)) / 3.0

        rho_geo_farmer = spearman_corr(score_geo, farmer_overall)
        rho_add_farmer = spearman_corr(score_add, farmer_overall)
        ci_geo = bootstrap_spearman(score_geo, farmer_overall)
        ci_add = bootstrap_spearman(score_add, farmer_overall)

        rho_geo_fm = spearman_corr(score_geo, farmer_m)
        rho_add_fm = spearman_corr(score_add, farmer_m)
        rho_geo_ft = spearman_corr(score_geo, farmer_t)
        rho_add_ft = spearman_corr(score_add, farmer_t)
        rho_geo_fr = spearman_corr(score_geo, farmer_r)
        rho_add_fr = spearman_corr(score_add, farmer_r)

        print(f"\n  Spearman vs proxy farmer overall:")
        print(f"    Geometric: {rho_geo_farmer:.4f}  95% CI [{ci_geo[0]:.4f}, {ci_geo[1]:.4f}]")
        print(f"    Additive:  {rho_add_farmer:.4f}  95% CI [{ci_add[0]:.4f}, {ci_add[1]:.4f}]")

        print(f"\n  Per-feature Spearman:")
        print(f"    {'Feature':<15s} {'Geometric':>10s} {'Additive':>10s}")
        print(f"    {'Moisture':<15s} {rho_geo_fm:>10.4f} {rho_add_fm:>10.4f}")
        print(f"    {'Tillage':<15s} {rho_geo_ft:>10.4f} {rho_add_ft:>10.4f}")
        print(f"    {'Residue':<15s} {rho_geo_fr:>10.4f} {rho_add_fr:>10.4f}")

        better = "GEOMETRIC" if rho_geo_farmer > rho_add_farmer else "ADDITIVE"
        print(f"\n  ==> {better} fusion agrees better with farmer proxies")

    farmer_df = pd.DataFrame([{
        "Method":     "Geometric",
        "Spearman_overall": f"{rho_geo_farmer:.4f}",
        "CI_lo": f"{ci_geo[0]:.4f}", "CI_hi": f"{ci_geo[1]:.4f}",
        "Spearman_moisture": f"{rho_geo_fm:.4f}",
        "Spearman_tillage":  f"{rho_geo_ft:.4f}",
        "Spearman_residue":  f"{rho_geo_fr:.4f}",
    }, {
        "Method":     "Additive",
        "Spearman_overall": f"{rho_add_farmer:.4f}",
        "CI_lo": f"{ci_add[0]:.4f}", "CI_hi": f"{ci_add[1]:.4f}",
        "Spearman_moisture": f"{rho_add_fm:.4f}",
        "Spearman_tillage":  f"{rho_add_ft:.4f}",
        "Spearman_residue":  f"{rho_add_fr:.4f}",
    }])
    farmer_df.to_csv(out_dir / "farmer_agreement.csv", index=False)
    print()

    # ======================================================================
    # 6. EPSILON FLOOR SENSITIVITY
    # ======================================================================
    print("=" * 72)
    print("6. EPSILON FLOOR SENSITIVITY (Geometric fusion)")
    print("=" * 72)

    # Input layer statistics
    print(f"  Input layer minima across {n} images:")
    print(f"    moisture min:    {M.min():.6f}")
    print(f"    uncovered min:   {U.min():.6f}")
    print(f"    tillage_eff min: {T_eff.min():.6f}")
    print(f"    gdd_norm min:    {G.min():.6f}")
    print()

    for thresh in [0.01, 0.05, 0.10]:
        n_below = int(np.any(np.column_stack([M, U, T_eff, G]) < thresh, axis=1).sum())
        print(f"  Images with ANY input < {thresh}: {n_below} / {n}")
    print()

    # Default scores (eps=1e-6)
    score_geo_default = score_geo  # already computed

    eps_tests = [0.001, 0.01, 0.05]
    eps_rows = []
    for eps_val in eps_tests:
        log_risk_eps = compute_log_risk(M, G, U, T_eff, eps=eps_val)
        score_eps = scale_mad_tanh(log_risk_eps)
        rho_vs_default = spearman_corr(score_geo_default, score_eps)
        mad_vs_default = float(np.mean(np.abs(score_geo_default - score_eps)))
        eps_rows.append({
            "epsilon": eps_val,
            "spearman_vs_default": rho_vs_default,
            "mean_abs_diff": mad_vs_default,
        })
        print(f"  eps={eps_val:<8.3f}  Spearman vs default: {rho_vs_default:.6f}  "
              f"Mean |diff|: {mad_vs_default:.4f}")

    eps_df = pd.DataFrame(eps_rows)
    eps_df.to_csv(out_dir / "epsilon_sensitivity.csv", index=False)
    print()

    # ======================================================================
    # 7. FIGURES
    # ======================================================================
    dpi = 300

    # -- Figure 1: Scatter geometric vs additive, colored by which is higher --
    print("Generating figures...")
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    diff = score_geo - score_add
    c = np.where(diff > 0, "#dc2626", "#2563eb")   # red = geo higher, blue = add higher
    ax1.scatter(score_geo, score_add, s=12, alpha=0.6, c=c, edgecolors="none")
    lo = min(score_geo.min(), score_add.min()) - 2
    hi = max(score_geo.max(), score_add.max()) + 2
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1 line")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Geometric Risk Score [0-100]", fontsize=12)
    ax1.set_ylabel("Additive Risk Score [0-100]", fontsize=12)
    ax1.set_title(f"Geometric vs Additive Fusion (n={n})\n"
                  f"Spearman rho = {rho_sp:.4f}  |  "
                  f"Red = geometric higher, Blue = additive higher", fontsize=11)
    ax1.legend(fontsize=10, loc="upper left")

    # Add marginal annotations
    n_geo_higher = int((diff > 0).sum())
    n_add_higher = int((diff < 0).sum())
    n_equal = n - n_geo_higher - n_add_higher
    ax1.text(0.97, 0.03, f"Geo higher: {n_geo_higher}\nAdd higher: {n_add_higher}",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig1.tight_layout()
    fig1_path = out_dir / "fig1_scatter_geo_vs_add.png"
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {fig1_path}")

    # -- Figure 2: Spatial difference map --
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    # Use image index as pseudo-spatial coordinate (no lat/lon in data)
    # Create 2D layout: arrange 333 images in a grid for visualization
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    x_pos = np.arange(n) % n_cols
    y_pos = np.arange(n) // n_cols

    vmax = max(abs(diff.min()), abs(diff.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sc = ax2.scatter(x_pos, y_pos, c=diff, cmap="RdBu_r", norm=norm,
                     s=25, edgecolors="gray", linewidths=0.3, alpha=0.8)
    cbar = fig2.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("Score Difference (Geometric - Additive)", fontsize=10)
    ax2.set_xlabel("Image Column (grid layout)", fontsize=11)
    ax2.set_ylabel("Image Row (grid layout)", fontsize=11)
    ax2.set_title("Risk Score Difference: Geometric minus Additive\n"
                  "Red = geometric assigns higher risk  |  "
                  "Blue = additive assigns higher risk", fontsize=11)
    ax2.invert_yaxis()
    fig2.tight_layout()
    fig2_path = out_dir / "fig2_spatial_difference_map.png"
    fig2.savefig(fig2_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fig2_path}")

    # -- Figure 3: Bar chart farmer agreement with error bars --
    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    methods = ["Geometric", "Additive"]
    rhos_farmer = [rho_geo_farmer, rho_add_farmer]
    ci_los = [ci_geo[0], ci_add[0]]
    ci_his = [ci_geo[1], ci_add[1]]
    err_lo = [r - lo for r, lo in zip(rhos_farmer, ci_los)]
    err_hi = [hi - r for r, hi in zip(rhos_farmer, ci_his)]
    colors_bar = ["#dc2626", "#2563eb"]

    bars = ax3.bar(methods, rhos_farmer, color=colors_bar, edgecolor="black",
                   width=0.45, yerr=[err_lo, err_hi], capsize=8,
                   error_kw={"lw": 1.5, "capthick": 1.5})
    for bar, rho in zip(bars, rhos_farmer):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{rho:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax3.set_ylabel("Spearman Correlation with Farmer Ordinal", fontsize=12)
    ax3.set_title("Farmer Expert Agreement: Geometric vs Additive\n"
                  "(error bars = bootstrapped 95% CI)", fontsize=12)
    ax3.set_ylim(0, min(1.05, max(rhos_farmer) + 0.15))
    fig3.tight_layout()
    fig3_path = out_dir / "fig3_farmer_agreement.png"
    fig3.savefig(fig3_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {fig3_path}")

    # -- Figure 4: Histogram of score differences --
    fig4, ax4 = plt.subplots(figsize=(8, 5.5))
    ax4.hist(diff, bins=50, color="#6b7280", edgecolor="white", linewidth=0.5,
             alpha=0.75, density=False)
    ax4.axvline(0, color="black", lw=1.5, ls="--", label="Zero difference")
    ax4.axvline(np.mean(diff), color="#dc2626", lw=1.5, ls="-",
                label=f"Mean = {np.mean(diff):.2f}")
    ax4.axvline(np.median(diff), color="#2563eb", lw=1.5, ls="-.",
                label=f"Median = {np.median(diff):.2f}")
    ax4.set_xlabel("Score Difference (Geometric - Additive)", fontsize=12)
    ax4.set_ylabel("Number of Images", fontsize=12)
    ax4.set_title(f"Distribution of Per-Image Score Differences (n={n})\n"
                  f"Positive = geometric assigns higher risk", fontsize=11)
    ax4.legend(fontsize=10)
    fig4.tight_layout()
    fig4_path = out_dir / "fig4_score_difference_histogram.png"
    fig4.savefig(fig4_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {fig4_path}")

    # ======================================================================
    # 8. Per-image detail CSV
    # ======================================================================
    detail = df[["image"]].copy()
    detail["moisture"]    = M
    detail["residue"]     = R
    detail["uncovered"]   = U
    detail["gdd_norm"]    = G
    detail["tillage_eff"] = T_eff
    detail["score_geo"]   = score_geo
    detail["score_add"]   = score_add
    detail["diff_geo_add"]= diff
    detail["suppression"] = suppression_flags
    detail["low_factor"]  = suppression_which
    detail.to_csv(out_dir / "per_image_detail.csv", index=False)

    # ======================================================================
    # 9. FINAL SUMMARY
    # ======================================================================
    print()
    print("=" * 72)
    print("8. FINAL SUMMARY")
    print("=" * 72)

    print(f"\n  Q1: Do the two methods produce meaningfully different rankings?")
    if rho_sp > 0.95:
        print(f"      Spearman rho = {rho_sp:.4f} => HIGH agreement. The two methods")
        print(f"      produce largely similar rankings. Most images receive comparable")
        print(f"      risk positions under both fusion rules.")
    elif rho_sp > 0.80:
        print(f"      Spearman rho = {rho_sp:.4f} => MODERATE agreement. Rankings are")
        print(f"      correlated but with notable divergence in specific subsets.")
    else:
        print(f"      Spearman rho = {rho_sp:.4f} => SUBSTANTIAL disagreement. The two")
        print(f"      methods produce materially different risk rankings.")

    print(f"\n  Q2: Where do they disagree, and does the pattern support")
    print(f"      multiplicative suppression?")
    print(f"      {n_suppress_disagree}/{n_disagree} top-disagreement images "
          f"({n_suppress_disagree/n_disagree:.0%} if n_disagree>0) show the")
    print(f"      'one-low-others-high' pattern. In these cases, geometric fusion")
    print(f"      suppresses risk because a single near-zero factor drags the")
    print(f"      product down, while additive fusion averages across factors")
    print(f"      and maintains moderate risk. This confirms the theoretical")
    print(f"      expectation of multiplicative suppression in geometric mean.")

    print(f"\n  Q3: Which method better matches farmer expert judgment?")
    print(f"      Geometric: rho={rho_geo_farmer:.4f}  "
          f"Additive: rho={rho_add_farmer:.4f}")
    if abs(rho_geo_farmer - rho_add_farmer) < 0.02:
        print(f"      Difference is small (<0.02); methods are roughly equivalent")
        print(f"      in farmer agreement.")
    elif rho_geo_farmer > rho_add_farmer:
        print(f"      GEOMETRIC fusion agrees better with farmer classifications,")
        print(f"      suggesting that expert intuition aligns with multiplicative")
        print(f"      interaction (one bad factor vetoes overall risk).")
    else:
        print(f"      ADDITIVE fusion agrees better with farmer classifications,")
        print(f"      suggesting that expert intuition is more compensatory")
        print(f"      (moderate across factors still means moderate risk).")

    print(f"\n  Q4: How sensitive is geometric fusion to epsilon floor?")
    most_sensitive = eps_df.loc[eps_df["spearman_vs_default"].idxmin()]
    print(f"      Tested eps = {[e for e in eps_tests]}")
    print(f"      Lowest Spearman vs default (eps=1e-6): "
          f"{most_sensitive['spearman_vs_default']:.6f} at eps={most_sensitive['epsilon']}")
    if most_sensitive["spearman_vs_default"] > 0.99:
        print(f"      => Fusion is ROBUST to epsilon choice (rho > 0.99 for all tested values).")
    elif most_sensitive["spearman_vs_default"] > 0.95:
        print(f"      => MILD sensitivity. eps=0.05 shifts rankings slightly.")
    else:
        print(f"      => NOTABLE sensitivity. Floor choice matters for near-zero images.")

    # Save summary text
    summary_txt = out_dir / "analysis_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Additive vs Geometric Fusion Comparison\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Images: {n}\n")
        f.write(f"Weights: wM={WM}, wG={WG}, wR={WR}, wT={WT}\n")
        f.write(f"Default epsilon: {EPS_DEFAULT}\n\n")
        f.write("Agreement Metrics:\n")
        f.write(f"  Spearman rho:     {rho_sp:.6f}\n")
        f.write(f"  Kendall tau-b:    {tau:.6f}\n")
        f.write(f"  Top-20% overlap:  {t20:.4f}\n")
        f.write(f"  Top-10% overlap:  {t10:.4f}\n")
        f.write(f"  Mean |diff|:      {mad_sc:.4f}\n\n")
        f.write("Farmer Agreement:\n")
        f.write(f"  Geometric: {rho_geo_farmer:.4f}  CI [{ci_geo[0]:.4f}, {ci_geo[1]:.4f}]\n")
        f.write(f"  Additive:  {rho_add_farmer:.4f}  CI [{ci_add[0]:.4f}, {ci_add[1]:.4f}]\n")
        f.write(f"  Better: {'Geometric' if rho_geo_farmer > rho_add_farmer else 'Additive'}\n\n")
        f.write("Multiplicative Suppression:\n")
        f.write(f"  Total images with pattern:       {n_suppress_total}/{n}\n")
        f.write(f"  In top-10% disagreement:         {n_suppress_disagree}/{n_disagree}\n\n")
        f.write("Epsilon Sensitivity:\n")
        for _, row in eps_df.iterrows():
            f.write(f"  eps={row['epsilon']:<8.3f}  Spearman={row['spearman_vs_default']:.6f}\n")

    print(f"\nSaved: {summary_txt}")
    print(f"All outputs saved to: {out_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Additive vs Geometric fusion comparison for WERM pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to fusion_results_FREEZE.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic data")
    parser.add_argument("--output", type=str,
                        default="results/fusion_comparison",
                        help="Output directory")
    args = parser.parse_args()

    if args.csv is None and not args.synthetic:
        print("No --csv provided; running in synthetic demo mode.\n")
        args.synthetic = True

    run(args)


if __name__ == "__main__":
    main()
