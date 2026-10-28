"""
Species-Specific Base-Temperature (Tb) Sensitivity Analysis
============================================================
Recomputes normalized GDD for three weed species with distinct Tb values,
re-runs the log-space weighted geometric mean fusion, and compares each
species-specific risk map against the original (Tb = 10 degC) pipeline.

Two comparison modes:
  1. "re-scaled" — each species gets its own MAD-tanh normalization
     (shows rank-order changes; for constant GDD, ranks are preserved)
  2. "fixed-scale" — applies the ORIGINAL centering/scale parameters
     to species risk vectors (shows actual absolute risk shift)

Usage (with frozen CSV from pipeline):
    python species_sensitivity.py --csv path/to/fusion_results_FREEZE.csv

Usage (with GDD CSV from NASA POWER API):
    python species_sensitivity.py --csv path/to/fusion_results_FREEZE.csv \
                                  --gdd-csv path/to/gdd.csv

Usage (standalone demo with synthetic data):
    python species_sensitivity.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Pipeline constants (mirrors fusion_final.py & temperature_gdd.py) ────────
ORIGINAL_TB       = 10.0          # current pipeline base temperature (degC)
T_BASELINE_GDD    = 120.0         # normalization denominator
WM, WG, WR, WT   = 0.30, 0.20, 0.20, 0.30   # fusion weights
EPS               = 1e-6
N_QUINTILES       = 5

# ── Species definitions ──────────────────────────────────────────────────────
SPECIES = [
    {"name": "Phalaris minor",     "common": "littleseed canary grass", "tb": 1.0},
    {"name": "Chenopodium album",  "common": "bathua",                 "tb": 3.0},
    {"name": "Melilotus indica",   "common": "senji",                  "tb": 0.0},
]


# ── Helper functions ─────────────────────────────────────────────────────────

def compute_log_risk(M, G, U, T_eff):
    """Weighted geometric mean fusion in log space."""
    Mc = np.maximum(np.asarray(M, dtype=float), EPS)
    Gc = np.maximum(np.asarray(G, dtype=float), EPS)
    Uc = np.maximum(np.asarray(U, dtype=float), EPS)
    Tc = np.maximum(np.asarray(T_eff, dtype=float), EPS)
    return (WM * np.log(Mc) + WG * np.log(Gc)
            + WR * np.log(Uc) + WT * np.log(Tc))


def scale_mad_tanh(log_risk):
    """Robust MAD-tanh scaling to [0, 100]  (matches fusion_final.py).
    Returns (scores, mu, sigma) so we can reuse parameters."""
    x = np.asarray(log_risk, dtype=float)
    med = float(np.median(x))
    mad_val = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad_val if mad_val > 1e-12 else (float(np.std(x)) + 1e-12)
    z = (x - med) / sigma
    scores = 50.0 + 50.0 * np.tanh(z / 2.0)
    return scores, med, sigma


def scale_mad_tanh_fixed(log_risk, mu, sigma):
    """Apply MAD-tanh scaling with pre-computed mu/sigma (fixed-scale mode)."""
    x = np.asarray(log_risk, dtype=float)
    z = (x - mu) / sigma
    return 50.0 + 50.0 * np.tanh(z / 2.0)


def spearman_corr(a, b):
    """Spearman rank correlation (no scipy dependency)."""
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    ar -= ar.mean(); br -= br.mean()
    denom = np.sqrt((ar ** 2).sum()) * np.sqrt((br ** 2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else float("nan")


def top_k_overlap(scores_orig, scores_new, frac=0.20):
    """Fraction of original top-frac% images that are also in new top-frac%."""
    k = int(np.ceil(frac * len(scores_orig)))
    top_orig = set(np.argsort(scores_orig)[-k:])
    top_new  = set(np.argsort(scores_new)[-k:])
    return len(top_orig & top_new) / len(top_orig) if len(top_orig) > 0 else 0.0


def quintile_labels(scores, n=N_QUINTILES):
    """Assign quintile labels Q1..Q5 (Q1 = lowest risk, Q5 = highest)."""
    qs = pd.qcut(pd.Series(scores).rank(method="first"),
                 q=n, labels=[f"Q{i+1}" for i in range(n)])
    return qs.to_numpy()


def quintile_migration_matrix(orig_q, new_q, n=N_QUINTILES):
    """5x5 matrix: rows = original quintile, cols = new quintile."""
    labels = [f"Q{i+1}" for i in range(n)]
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for o, nq in zip(orig_q, new_q):
        mat.loc[o, nq] += 1
    return mat


def recompute_gdd_norm(daily_temps, tb):
    """GDD = sum(max(Tavg - Tb, 0)); normalize by T_BASELINE_GDD."""
    gdd = sum(max(t - tb, 0.0) for t in daily_temps)
    return min(gdd / T_BASELINE_GDD, 1.0), gdd


def generate_synthetic_data(n=333, seed=42):
    """Generate representative synthetic fusion data for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image":        [f"img_{i:04d}.jpg" for i in range(n)],
        "moisture":     rng.beta(2.5, 2.5, n),
        "residue":      rng.beta(2.0, 5.0, n),
        "gdd_norm":     np.full(n, 0.45),           # constant (single-date)
        "tillage_norm": rng.beta(3.0, 3.0, n),
        "tillage_eff":  None,                        # will be computed
    })


def generate_synthetic_gdd_csv(seed=42):
    """Synthetic 7-day temperature series near Indore, India winter."""
    rng = np.random.default_rng(seed)
    tmax = rng.normal(26.0, 2.0, 7)
    tmin = rng.normal(12.0, 2.0, 7)
    tavg = (tmax + tmin) / 2.0
    return pd.DataFrame({
        "date":          [f"2026020{i+1}" for i in range(7)],
        "T2M_MAX_C":     tmax,
        "T2M_MIN_C":     tmin,
        "Tavg_C":        tavg,
        "Daily_GDD_Cd":  np.maximum(tavg - ORIGINAL_TB, 0.0),
    })


# ── Main analysis ────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load or generate data ─────────────────────────────────────────────
    synthetic_mode = args.synthetic or (args.csv is None)

    if synthetic_mode:
        print("=== Running in SYNTHETIC mode (no real pipeline CSV) ===\n")
        df = generate_synthetic_data()
        gdd_df = generate_synthetic_gdd_csv()
        df["tillage_eff"] = df["tillage_norm"] * (1.0 - df["residue"])
    else:
        df = pd.read_csv(args.csv)
        needed = ["image", "moisture", "residue", "gdd_norm", "tillage_eff"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: CSV missing columns: {missing}\nFound: {list(df.columns)}")
        gdd_df = pd.read_csv(args.gdd_csv) if args.gdd_csv else None

    n = len(df)
    M       = df["moisture"].to_numpy(dtype=float)
    R       = df["residue"].to_numpy(dtype=float)
    U       = 1.0 - R
    T_eff   = df["tillage_eff"].to_numpy(dtype=float)
    G_orig  = df["gdd_norm"].to_numpy(dtype=float)

    gdd_is_constant = np.std(G_orig) < 1e-9
    g_orig_val = float(G_orig[0]) if gdd_is_constant else None

    print(f"Images:              {n}")
    if gdd_is_constant:
        print(f"GDD constant:        True  (G_norm = {g_orig_val:.6f})")
    else:
        print(f"GDD varies:          min={G_orig.min():.4f}  max={G_orig.max():.4f}")
    print(f"Original Tb:         {ORIGINAL_TB} degC")
    print(f"Normalization:       G_norm = min(cumGDD / {T_BASELINE_GDD}, 1.0)\n")

    # ── 2. Daily temperatures for species Tb re-computation ──────────────────
    if gdd_df is not None:
        daily_tavg = gdd_df["Tavg_C"].to_numpy(dtype=float)
        print(f"Daily Tavg from GDD CSV ({len(daily_tavg)} days):")
        print(f"  {daily_tavg.round(2).tolist()}\n")
    else:
        total_orig = g_orig_val * T_BASELINE_GDD if g_orig_val else G_orig.mean() * T_BASELINE_GDD
        print(f"No GDD CSV; estimated original total GDD = {total_orig:.2f} Cd\n")
        daily_tavg = None

    # ── 3. Original risk scores ──────────────────────────────────────────────
    log_risk_orig = compute_log_risk(M, G_orig, U, T_eff)
    risk_raw_orig = np.exp(log_risk_orig)
    score_orig, mu_orig, sigma_orig = scale_mad_tanh(log_risk_orig)
    q_orig = quintile_labels(score_orig)

    print(f"Original scaling: mu={mu_orig:.6f}, sigma={sigma_orig:.6f}")
    print(f"Original risk_raw: mean={risk_raw_orig.mean():.6f}, "
          f"median={np.median(risk_raw_orig):.6f}\n")

    # ── 4. Species-specific re-computation ───────────────────────────────────
    results = []
    species_scores_rescaled = {}   # each species gets own MAD-tanh
    species_scores_fixed    = {}   # uses original mu/sigma
    species_risk_raw        = {}
    species_gnorm           = {}

    for sp in SPECIES:
        name, tb = sp["name"], sp["tb"]
        print(f"--- {name}  (Tb = {tb} degC) ---")

        # Recompute GDD
        if daily_tavg is not None:
            g_norm_sp, gdd_total_sp = recompute_gdd_norm(daily_tavg, tb)
        else:
            gdd_total_orig_val = (g_orig_val if g_orig_val else G_orig.mean()) * T_BASELINE_GDD
            gdd_total_sp = max(gdd_total_orig_val + (ORIGINAL_TB - tb) * 7.0, 0.0)
            g_norm_sp = min(gdd_total_sp / T_BASELINE_GDD, 1.0)

        species_gnorm[name] = g_norm_sp
        print(f"  Total GDD:      {gdd_total_sp:.2f} Cd")
        print(f"  G_norm:         {g_norm_sp:.6f}")

        G_sp = np.full(n, g_norm_sp) if gdd_is_constant else np.clip(G_orig * (g_norm_sp / (g_orig_val + 1e-12)), 0.0, 1.0)

        # Fusion
        log_risk_sp = compute_log_risk(M, G_sp, U, T_eff)
        rr_sp = np.exp(log_risk_sp)
        species_risk_raw[name] = rr_sp

        # Mode 1: independent MAD-tanh (re-scaled)
        score_sp_re, _, _ = scale_mad_tanh(log_risk_sp)
        species_scores_rescaled[name] = score_sp_re

        # Mode 2: fixed-scale (use original mu/sigma)
        score_sp_fix = scale_mad_tanh_fixed(log_risk_sp, mu_orig, sigma_orig)
        species_scores_fixed[name] = score_sp_fix

        q_sp_re  = quintile_labels(score_sp_re)
        q_sp_fix = quintile_labels(score_sp_fix)

        # Metrics (re-scaled)
        rho_re    = spearman_corr(score_orig, score_sp_re)
        top20_re  = top_k_overlap(score_orig, score_sp_re)
        mad_re    = float(np.mean(np.abs(score_orig - score_sp_re)))
        mig_re    = quintile_migration_matrix(q_orig, q_sp_re)

        # Metrics (fixed-scale)
        rho_fix   = spearman_corr(score_orig, score_sp_fix)
        top20_fix = top_k_overlap(score_orig, score_sp_fix)
        mad_fix   = float(np.mean(np.abs(score_orig - score_sp_fix)))
        mean_fix  = float(np.mean(score_sp_fix - score_orig))
        mig_fix   = quintile_migration_matrix(q_orig, q_sp_fix)

        # Raw risk metrics
        raw_ratio = float(np.mean(rr_sp / risk_raw_orig))
        raw_mean_diff = float(np.mean(rr_sp - risk_raw_orig))

        results.append({
            "species": name, "common_name": sp["common"], "tb_degC": tb,
            "gdd_total_Cd": gdd_total_sp, "gdd_norm": g_norm_sp,
            # re-scaled metrics
            "rho_rescaled": rho_re, "top20_rescaled": top20_re,
            "mad_rescaled": mad_re, "mig_rescaled": mig_re,
            # fixed-scale metrics
            "rho_fixed": rho_fix, "top20_fixed": top20_fix,
            "mad_fixed": mad_fix, "mean_signed_fixed": mean_fix,
            "mig_fixed": mig_fix,
            # raw
            "raw_mean_ratio": raw_ratio, "raw_mean_diff": raw_mean_diff,
        })

        print(f"  Raw risk ratio:   {raw_ratio:.6f}x  (species/original)")
        print(f"  [Re-scaled] Spearman={rho_re:.6f}  Top20={top20_re:.4f}  MAD={mad_re:.4f}")
        print(f"  [Fixed-scl] Spearman={rho_fix:.6f}  Top20={top20_fix:.4f}  "
              f"MAD={mad_fix:.4f}  Mean diff={mean_fix:+.4f}")
        print()

    # ── 5. Output tables ─────────────────────────────────────────────────────
    # 5a) Summary table
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Species":                      r["species"],
            "Common Name":                  r["common_name"],
            "Tb (degC)":                    r["tb_degC"],
            "GDD Total (Cd)":              f'{r["gdd_total_Cd"]:.2f}',
            "GDD Normalized":              f'{r["gdd_norm"]:.6f}',
            "Raw Risk Ratio (sp/orig)":    f'{r["raw_mean_ratio"]:.6f}',
            "Spearman rho [re-scaled]":    f'{r["rho_rescaled"]:.6f}',
            "Top-20% Overlap [re-scaled]": f'{r["top20_rescaled"]:.4f}',
            "Mean |Diff| [re-scaled]":     f'{r["mad_rescaled"]:.4f}',
            "Spearman rho [fixed-scale]":  f'{r["rho_fixed"]:.6f}',
            "Top-20% Overlap [fixed-scl]": f'{r["top20_fixed"]:.4f}',
            "Mean |Diff| [fixed-scale]":   f'{r["mad_fixed"]:.4f}',
            "Mean Signed Diff [fixed]":    f'{r["mean_signed_fixed"]:+.4f}',
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "species_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    # 5b) Quintile migration matrices
    for r in results:
        tag = r["species"].replace(" ", "_")
        for mode in ["rescaled", "fixed"]:
            mig_path = out_dir / f"migration_matrix_{tag}_{mode}.csv"
            r[f"mig_{mode}"].to_csv(mig_path)
        print(f"\n--- Quintile Migration (fixed-scale): {r['species']} (Tb={r['tb_degC']}) ---")
        print(r["mig_fixed"].to_string())
        print(f"Saved: {out_dir / f'migration_matrix_{tag}_fixed.csv'}")

    # 5c) Per-image detail CSV
    detail = df[["image"]].copy()
    detail["risk_raw_original"] = risk_raw_orig
    detail["score_original"] = score_orig
    detail["quintile_original"] = q_orig
    for r in results:
        tag = r["species"].replace(" ", "_")
        detail[f"risk_raw_{tag}"] = species_risk_raw[r["species"]]
        detail[f"score_rescaled_{tag}"] = species_scores_rescaled[r["species"]]
        detail[f"score_fixed_{tag}"] = species_scores_fixed[r["species"]]
        detail[f"quintile_fixed_{tag}"] = quintile_labels(species_scores_fixed[r["species"]])
    detail_path = out_dir / "per_image_scores.csv"
    detail.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}")

    # ── 6. Figures (all use fixed-scale for meaningful comparison) ────────────
    dpi = 300
    colors = ["#dc2626", "#2563eb", "#16a34a"]

    # Figure 1: Three-panel scatter (fixed-scale)
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, r, c in zip(axes, results, colors):
        sp_name = r["species"]
        sp_sc = species_scores_fixed[sp_name]
        ax.scatter(score_orig, sp_sc, s=8, alpha=0.5, edgecolors="none", c=c)
        lo = min(score_orig.min(), sp_sc.min()) - 2
        hi = max(score_orig.max(), sp_sc.max()) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1 line")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Original Risk Score (Tb=10)", fontsize=10)
        ax.set_title(f"{sp_name}\n(Tb = {r['tb_degC']} degC)", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Species-Specific Risk Score (fixed-scale)", fontsize=10)
    fig1.suptitle("Species Tb Sensitivity: Original vs Species Risk Scores",
                  fontsize=13, y=1.02)
    fig1.tight_layout()
    fig1_path = out_dir / "fig1_scatter_species_vs_original.png"
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {fig1_path}")

    # Figure 2: Overlaid risk score distributions
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    bins_hist = np.linspace(0, 100, 51)
    ax2.hist(score_orig, bins=bins_hist, alpha=0.35, density=True,
             label=f"Original (Tb={ORIGINAL_TB})", color="#6b7280",
             edgecolor="white", linewidth=0.5)
    for r, color in zip(results, colors):
        sp_sc = species_scores_fixed[r["species"]]
        ax2.hist(sp_sc, bins=bins_hist, alpha=0.25, density=True,
                 label=f'{r["species"]} (Tb={r["tb_degC"]})',
                 color=color, edgecolor="white", linewidth=0.5)
    # Smoothed KDE overlay
    for label_name, scores, color in [
        (f"Original (Tb={ORIGINAL_TB})", score_orig, "#6b7280"),
        *[(f'{r["species"]} (Tb={r["tb_degC"]})', species_scores_fixed[r["species"]], c)
          for r, c in zip(results, colors)]
    ]:
        counts, edges = np.histogram(scores, bins=100, range=(0, 100), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        kernel = np.ones(5) / 5
        smooth = np.convolve(counts, kernel, mode="same")
        ax2.plot(centers, smooth, color=color, lw=1.8)
    ax2.set_xlabel("Risk Score [0-100] (fixed-scale)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Risk Score Distributions: Original vs Species-Specific Tb",
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 100)
    fig2.tight_layout()
    fig2_path = out_dir / "fig2_score_distributions.png"
    fig2.savefig(fig2_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # Figure 3: Bar chart of Spearman correlations (both modes)
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    names = [r["species"] for r in results]
    tbs   = [r["tb_degC"] for r in results]
    bar_labels = [f"{n}\n(Tb={t})" for n, t in zip(names, tbs)]

    # Re-scaled
    rhos_re = [r["rho_rescaled"] for r in results]
    bars_re = ax3a.bar(bar_labels, rhos_re, color=colors, edgecolor="black", width=0.5)
    ax3a.set_ylabel("Spearman rho vs Original", fontsize=11)
    ax3a.set_title("Re-scaled Mode\n(each species independently normalized)", fontsize=11)
    ax3a.set_ylim(min(0, min(rhos_re) - 0.05), 1.08)
    ax3a.axhline(1.0, color="gray", ls="--", lw=0.8)
    for bar, rho in zip(bars_re, rhos_re):
        ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{rho:.4f}", ha="center", va="bottom", fontsize=10)

    # Fixed-scale
    rhos_fix = [r["rho_fixed"] for r in results]
    bars_fix = ax3b.bar(bar_labels, rhos_fix, color=colors, edgecolor="black", width=0.5)
    ax3b.set_ylabel("Spearman rho vs Original", fontsize=11)
    ax3b.set_title("Fixed-scale Mode\n(original normalization parameters)", fontsize=11)
    ax3b.set_ylim(min(0, min(rhos_fix) - 0.05), 1.08)
    ax3b.axhline(1.0, color="gray", ls="--", lw=0.8)
    for bar, rho in zip(bars_fix, rhos_fix):
        ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{rho:.4f}", ha="center", va="bottom", fontsize=10)

    fig3.suptitle("Rank Correlation with Original Risk Map (Tb=10 degC)", fontsize=13, y=1.02)
    fig3.tight_layout()
    fig3_path = out_dir / "fig3_spearman_correlations.png"
    fig3.savefig(fig3_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {fig3_path}")

    # ── 7. GDD comparison table ──────────────────────────────────────────────
    gdd_table_rows = [
        {"Species": "Original (pipeline)", "Tb (degC)": ORIGINAL_TB,
         "Total GDD (Cd)": g_orig_val * T_BASELINE_GDD if g_orig_val else float("nan"),
         "G_norm": g_orig_val if g_orig_val else float("nan"),
         "Delta GDD vs Orig": 0.0}
    ]
    for r in results:
        orig_total = g_orig_val * T_BASELINE_GDD if g_orig_val else float("nan")
        gdd_table_rows.append({
            "Species": r["species"],
            "Tb (degC)": r["tb_degC"],
            "Total GDD (Cd)": r["gdd_total_Cd"],
            "G_norm": r["gdd_norm"],
            "Delta GDD vs Orig": r["gdd_total_Cd"] - orig_total if not np.isnan(orig_total) else float("nan"),
        })
    gdd_table = pd.DataFrame(gdd_table_rows)
    gdd_path = out_dir / "gdd_comparison.csv"
    gdd_table.to_csv(gdd_path, index=False)

    print(f"\n{'='*78}")
    print("GDD COMPARISON TABLE")
    print(f"{'='*78}")
    print(gdd_table.to_string(index=False))
    print(f"Saved: {gdd_path}")

    # ── 8. Constant-GDD special note ─────────────────────────────────────────
    if gdd_is_constant:
        print(f"\n{'='*78}")
        print("NOTE: CONSTANT GDD ACROSS ALL IMAGES (single-date collection)")
        print(f"{'='*78}")
        print(f"Original: Tb={ORIGINAL_TB}, G_norm={g_orig_val:.6f}, "
              f"Total GDD={g_orig_val * T_BASELINE_GDD:.2f} Cd")
        print()
        print("Because GDD is identical for ALL images:")
        print("  - Changing Tb shifts log(risk) by a constant delta = wG * log(G_sp/G_orig)")
        print("  - Re-scaled mode: MAD-tanh is location-invariant => Spearman rho = 1.0")
        print("    (rankings unchanged, scores identical after re-normalization)")
        print("  - Fixed-scale mode: the constant shift produces a uniform score offset")
        print("    (Spearman rho = 1.0 but absolute scores shift up/down)")
        print()
        print("The meaningful comparison is in ABSOLUTE GDD values:")
        for r in results:
            delta = r["gdd_total_Cd"] - g_orig_val * T_BASELINE_GDD
            print(f"  {r['species']:25s}  Tb={r['tb_degC']:4.1f}  =>  "
                  f"GDD={r['gdd_total_Cd']:7.2f} Cd  (delta={delta:+.2f} Cd)")
        print()
        print("Lower Tb => more thermal accumulation => higher weed emergence pressure.")
        print("All three species have Tb < 10, accumulating GDD on days the original model ignores.")

    # ── 9. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("FINAL SUMMARY")
    print(f"{'='*78}")
    max_div_fix = max(results, key=lambda r: r["mad_fixed"])
    direction = "HIGHER" if max_div_fix["mean_signed_fixed"] > 0 else "LOWER"
    print(f"Largest divergence (fixed-scale): {max_div_fix['species']} "
          f"(Tb = {max_div_fix['tb_degC']} degC)")
    print(f"  Mean |diff| in risk score: {max_div_fix['mad_fixed']:.4f}")
    print(f"  Direction: {direction} average risk scores "
          f"(mean signed diff = {max_div_fix['mean_signed_fixed']:+.4f})")
    print(f"  Spearman rho: {max_div_fix['rho_fixed']:.6f}")

    # Largest raw risk change
    max_raw = max(results, key=lambda r: abs(r["raw_mean_ratio"] - 1.0))
    print(f"\nLargest raw risk multiplier: {max_raw['species']} "
          f"({max_raw['raw_mean_ratio']:.4f}x original)")

    # Save text summary
    summary_txt = out_dir / "analysis_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Species-Specific Tb Sensitivity Analysis\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Original pipeline: Tb = {ORIGINAL_TB} degC\n")
        f.write(f"Normalization: G_norm = min(cumGDD / {T_BASELINE_GDD}, 1.0)\n")
        f.write(f"Fusion: risk = exp(0.30*log(M) + 0.20*log(G) + 0.20*log(1-R) + 0.30*log(T_eff))\n")
        f.write(f"Scaling: MAD-tanh -> [0, 100]\n")
        f.write(f"Images: {n}\n")
        f.write(f"GDD constant across images: {gdd_is_constant}\n\n")
        f.write("Species Results (fixed-scale comparison):\n")
        f.write("-" * 55 + "\n")
        for r in results:
            f.write(f"\n{r['species']} ({r['common_name']})  Tb = {r['tb_degC']} degC\n")
            f.write(f"  GDD total:            {r['gdd_total_Cd']:.2f} Cd\n")
            f.write(f"  GDD normalized:       {r['gdd_norm']:.6f}\n")
            f.write(f"  Raw risk ratio:       {r['raw_mean_ratio']:.6f}x\n")
            f.write(f"  Spearman rho [fixed]: {r['rho_fixed']:.6f}\n")
            f.write(f"  Top-20%% overlap:      {r['top20_fixed']:.4f}\n")
            f.write(f"  Mean |diff| [fixed]:  {r['mad_fixed']:.4f}\n")
            f.write(f"  Mean signed [fixed]:  {r['mean_signed_fixed']:+.4f}\n")
        f.write(f"\nLargest divergence: {max_div_fix['species']} ({direction} risk)\n")
        f.write(f"Largest raw multiplier: {max_raw['species']} ({max_raw['raw_mean_ratio']:.4f}x)\n")
    print(f"\nSaved: {summary_txt}")
    print(f"All outputs saved to: {out_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Species-specific Tb sensitivity analysis for WERM pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to fusion_results_FREEZE.csv (frozen baseline)")
    parser.add_argument("--gdd-csv", type=str, default=None,
                        help="Path to gdd.csv with daily temperatures")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic data (no pipeline CSV needed)")
    parser.add_argument("--output", type=str,
                        default="results/species_sensitivity",
                        help="Output directory")
    args = parser.parse_args()

    if args.csv is None and not args.synthetic:
        print("No --csv provided; running in synthetic demo mode.")
        print("Use --csv path/to/fusion_results_FREEZE.csv for real data.\n")
        args.synthetic = True

    run(args)


if __name__ == "__main__":
    main()
