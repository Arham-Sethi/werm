"""
WERM Paper Figures — Publication-Quality Generator (v3)
=======================================================
Generates all main + supplementary figures for the WERM fusion paper.
Same data roots as v2, but with professional research-paper aesthetics:

  - Nature/Science-inspired minimalist styling
  - Colorblind-safe palette (Okabe-Ito + curated accents)
  - Combined multi-panel layouts (F4, F9) to save page space
  - Violin + strip overlays instead of plain boxplots
  - KDE overlays on histograms
  - Marginal distributions on scatter plots
  - Proper (a)/(b) panel labels
  - Journal single-column (3.5 in) and double-column (7.2 in) widths
  - 300 DPI vector-quality PNG + optional PDF export
"""

import os
import glob
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
ROOT = "/content/drive/MyDrive/WERM"
EXP  = os.path.join(ROOT, "experiments")
OUT  = os.path.join(ROOT, "paper_figures_v3")
os.makedirs(OUT, exist_ok=True)

SAVE_PDF = True  # set False to skip PDF companion files

# Baseline fusion settings
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30
eps = 1e-6

# ════════════════════════════════════════════════════════════════════
# COLOR SYSTEM  (Okabe-Ito core + muted accents, colorblind-safe)
# ════════════════════════════════════════════════════════════════════
PAL = {
    # Okabe-Ito primaries
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "green":    "#009E73",
    "vermil":   "#D55E00",
    "skyblue":  "#56B4E9",
    "pink":     "#CC79A7",
    "yellow":   "#F0E442",
    # Neutrals
    "black":    "#1B1B1B",
    "dgray":    "#3D3D3D",
    "mgray":    "#7A7A7A",
    "lgray":    "#BFBFBF",
    "vlgray":   "#E8E8E8",
    "white":    "#FFFFFF",
}

# Sequential series for up to 6 categories
SERIES = [PAL["blue"], PAL["orange"], PAL["green"],
          PAL["vermil"], PAL["skyblue"], PAL["pink"]]

# Risk classes: A (low) -> E (high)
CLS_COLORS = {
    "A": "#3A86A8",  # steel blue
    "B": "#5FAD56",  # leaf green
    "C": "#F2C14E",  # amber
    "D": "#F78154",  # tangerine
    "E": "#D1495B",  # crimson
}

# Custom sequential colormap for risk scores
_risk_colors = ["#F7FBFF", "#6BAED6", "#2171B5", "#08306B"]
RISK_CMAP = LinearSegmentedColormap.from_list("werm_risk", _risk_colors, N=256)

# Custom diverging colormap for disagreement
_div_colors = ["#2166AC", "#F7F7F7", "#B2182B"]
DIV_CMAP = LinearSegmentedColormap.from_list("werm_div", _div_colors, N=256)


# ════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ════════════════════════════════════════════════════════════════════
def setup_style():
    mpl.rcParams.update({
        # Resolution
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Spines
        "axes.edgecolor": PAL["dgray"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7.5,
        # Ticks
        "xtick.color": PAL["dgray"],
        "ytick.color": PAL["dgray"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Grid
        "axes.grid": False,  # we add grids selectively
        "grid.color": PAL["vlgray"],
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        # Math
        "mathtext.fontset": "dejavusans",
    })


# ════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ════════════════════════════════════════════════════════════════════
# Standard journal widths (inches)
COL1 = 3.5    # single column
COL15 = 5.5   # 1.5 column
COL2 = 7.2    # double column


def _save(fig, name):
    """Save PNG (always) + PDF (optional)."""
    base = os.path.join(OUT, name)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    if SAVE_PDF:
        fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.png" + (" + .pdf" if SAVE_PDF else ""))


def _panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold (a), (b), ... panel labels in axes coordinates."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _add_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, linewidth=0.5, color=PAL["vlgray"])


def _stat_annotation(ax, lines, x=0.03, y=0.97, ha="left", va="top"):
    """Small stat box without a visible border."""
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
            fontsize=6.5, family="monospace", color=PAL["dgray"],
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PAL["lgray"], alpha=0.85, linewidth=0.5))


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _kde_on_hist(ax, data, color, bw_method=0.3):
    """Overlay a smooth KDE curve on a histogram axis."""
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 300)
    try:
        kde = gaussian_kde(data, bw_method=bw_method)
        ys = kde(xs)
        # scale KDE to match histogram area
        hist_area = len(data) * (xmax - xmin) / 20  # approx bin width * n
        ys_scaled = ys * len(data) * (xmax - xmin) / 20
        ax.plot(xs, ys_scaled, color=color, linewidth=1.5, alpha=0.85)
    except Exception:
        pass  # silently skip if KDE fails


# ════════════════════════════════════════════════════════════════════
# DATA HELPERS  (unchanged from v2)
# ════════════════════════════════════════════════════════════════════
def find_one(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            hits.sort()
            return hits[0]
    return None


def find_in_exp(exp_name, patterns):
    base = os.path.join(EXP, exp_name)
    pats = [os.path.join(base, "**", p) for p in patterns]
    return find_one(pats)


def read_csv(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns: {list(df.columns)[:40]}")


def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    m = mad(x)
    if m == 0:
        m = 1e-12
    z = (x - med) / m
    return 50.0 * (np.tanh(z) + 1.0)


def risk_class(score):
    bins = [-1e9, 20, 40, 60, 80, 1e9]
    labels = ["A", "B", "C", "D", "E"]
    return pd.cut(score, bins=bins, labels=labels, right=False)


def spearman_corr(a, b):
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def top20_jaccard(a, b):
    k = int(0.2 * len(a))
    A = set(np.argsort(a)[-k:])
    B = set(np.argsort(b)[-k:])
    return len(A & B) / len(A | B)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
setup_style()

print("=" * 60)
print("  WERM Paper Figures v3 — Publication Quality")
print("=" * 60)

# ---------- locate data files ----------
risk_base_path    = find_in_exp("00_freeze", ["risk_eps_1e-6.csv", "*risk*1e-6*.csv", "*risk*eps*.csv"])
freeze_path       = find_in_exp("00_freeze", ["fusion_results_FREEZE.csv", "*FREEZE*.csv"])
var_alpha_path    = find_in_exp("01_residue_damping", ["variance_alpha0_alpha1.csv"])
alpha_sweep_path  = find_in_exp("01_residue_damping", ["alpha_sweep_summary.csv"])
step2_row_path    = find_in_exp("02_fusion_baselines", ["step2_scores_geo_add.csv", "*geo*add*.csv"])
step2_summary_path = find_in_exp("02_fusion_baselines", ["*summary*.csv"])
step3_weights_path = find_in_exp("03_weight_robustness", ["step3_weight_robustness_summary.csv", "*robustness*summary*.csv"])
step4_lofo_path   = find_in_exp("04_lofo_ablation", ["step4_lofo_summary.csv", "*lofo*summary*.csv"])
step5_scaler_path = find_in_exp("05_scaling_comparison", ["step5_scaler_summary.csv", "*scaler*summary*.csv"])
step6_noise_path  = find_in_exp("06_noise_robustness", ["step6_noise_summary.csv", "*noise*summary*.csv"])
step7_rank_stats_path = find_in_exp("07_dirichlet_weights", ["step7_image_rank_stats.csv"])
step7_dir_summary_path = find_in_exp("07_dirichlet_weights", ["step7_dirichlet_summary.csv", "*dirichlet*summary*.csv"])

print("\nData files located:")
for label, p in [
    ("baseline risk", risk_base_path), ("freeze", freeze_path),
    ("var_alpha", var_alpha_path), ("alpha_sweep", alpha_sweep_path),
    ("step2_row", step2_row_path), ("step3_weights", step3_weights_path),
    ("step4_lofo", step4_lofo_path), ("step5_scaler", step5_scaler_path),
    ("step6_noise", step6_noise_path), ("step7_rank", step7_rank_stats_path),
    ("step7_dir", step7_dir_summary_path),
]:
    status = "OK" if p else "MISSING"
    print(f"  [{status:>7s}] {label}: {p}")
print()

# ---------- ensure baseline risk ----------
df_risk = None
if risk_base_path is not None:
    df_risk = read_csv(risk_base_path)
else:
    if freeze_path is None:
        raise FileNotFoundError(
            "Need either risk_eps_1e-6.csv or fusion_results_FREEZE.csv"
        )
    print("Recomputing baseline risk from FREEZE ...")
    df_freeze = read_csv(freeze_path)
    id_col = get_col(df_freeze, ["image", "image_id", "img", "filename"])
    m_col  = get_col(df_freeze, ["moisture", "M"])
    r_col  = get_col(df_freeze, ["residue", "R"])
    t_col  = get_col(df_freeze, ["tillage_eff", "T_eff", "T"])
    g_col  = get_col(df_freeze, ["gdd_norm", "G"])
    M = df_freeze[m_col].astype(float).to_numpy()
    R = df_freeze[r_col].astype(float).to_numpy()
    U = 1.0 - R
    T = df_freeze[t_col].astype(float).to_numpy()
    G = df_freeze[g_col].astype(float).to_numpy()
    log_risk = (wM * np.log(np.maximum(M, eps)) + wG * np.log(np.maximum(G, eps))
                + wR * np.log(np.maximum(U, eps)) + wT * np.log(np.maximum(T, eps)))
    risk_raw = np.exp(log_risk)
    score = robust_scale(log_risk)
    cls = risk_class(score).astype(str)
    df_risk = pd.DataFrame({
        "image": df_freeze[id_col].values,
        "M": M, "R": R, "U": U, "T_eff": T, "G": G,
        "risk_raw": risk_raw, "log_risk": log_risk,
        "risk_score_0_100": score, "risk_class": cls,
    })
    out_tables = os.path.join(EXP, "00_freeze", "tables")
    os.makedirs(out_tables, exist_ok=True)
    risk_base_path = os.path.join(out_tables, "risk_eps_1e-6.csv")
    df_risk.to_csv(risk_base_path, index=False)
    print(f"  -> Saved recomputed baseline: {risk_base_path}")

score_col = get_col(df_risk, ["risk_score_0_100", "risk_score", "score"])
class_col = get_col(df_risk, ["risk_class", "class"])
id_col    = get_col(df_risk, ["image", "image_id"])

risk_scores = df_risk[score_col].astype(float).to_numpy()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Risk Score Distribution (histogram + KDE + rug)     ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F1: Risk score distribution")
fig, ax = plt.subplots(figsize=(COL2, 3.6), constrained_layout=True)

n_bins = 30
n, bins, patches = ax.hist(risk_scores, bins=n_bins, color=PAL["blue"],
                           alpha=0.65, edgecolor="white", linewidth=0.5)

# KDE overlay
xs = np.linspace(risk_scores.min() - 5, risk_scores.max() + 5, 400)
kde = gaussian_kde(risk_scores, bw_method=0.25)
ys = kde(xs) * len(risk_scores) * (bins[1] - bins[0])
ax.plot(xs, ys, color=PAL["vermil"], linewidth=1.6, label="KDE")

# Rug plot
ax.plot(risk_scores, np.full_like(risk_scores, -max(n) * 0.025), "|",
        color=PAL["blue"], alpha=0.15, markersize=4, markeredgewidth=0.5)

# Median + IQR lines
med = np.median(risk_scores)
q25, q75 = np.percentile(risk_scores, [25, 75])
ymax = ax.get_ylim()[1]
ax.axvline(med, color=PAL["vermil"], linewidth=1.0, linestyle="--", alpha=0.8)
ax.axvspan(q25, q75, color=PAL["blue"], alpha=0.06)
ax.text(med + 1, ymax * 0.92, f"median = {med:.1f}", fontsize=6.5,
        color=PAL["vermil"], va="top")

_add_grid(ax, "y")
ax.set_xlabel("Risk score (0\u2013100)")
ax.set_ylabel("Frequency")
ax.set_title("Baseline Risk Score Distribution")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

_stat_annotation(ax, [
    f"n = {len(risk_scores)}",
    f"mean  = {risk_scores.mean():.1f}",
    f"IQR   = [{q25:.1f}, {q75:.1f}]",
    f"std   = {risk_scores.std():.1f}",
], x=0.97, y=0.97, ha="right")

ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(bins[0] - 2, bins[-1] + 2)
ax.set_ylim(bottom=-max(n) * 0.04)
_save(fig, "F1_risk_distribution")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Risk Class Composition (horizontal bars + pct)      ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F2: Risk class composition")
counts = df_risk[class_col].value_counts()
order = ["A", "B", "C", "D", "E"]
vals = np.array([int(counts.get(k, 0)) for k in order])
total = vals.sum()
pcts = 100.0 * vals / total

fig, ax = plt.subplots(figsize=(COL2, 2.8), constrained_layout=True)

bars = ax.barh(order[::-1], vals[::-1],
               color=[CLS_COLORS[k] for k in order[::-1]],
               edgecolor="white", linewidth=0.6, height=0.62)

for bar, val, pct in zip(bars, vals[::-1], pcts[::-1]):
    x_pos = bar.get_width() + total * 0.012
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)", va="center", fontsize=7, color=PAL["dgray"])

ax.set_xlabel("Image count")
ax.set_title("Baseline Risk Class Composition")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
_add_grid(ax, "x")
ax.invert_yaxis()
# Subtle class label formatting
for tick in ax.get_yticklabels():
    cls = tick.get_text()
    tick.set_fontweight("bold")
    tick.set_color(CLS_COLORS.get(cls, PAL["dgray"]))

_save(fig, "F2_class_composition")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Variance Decomposition (alpha=1), lollipop chart    ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F3: Variance decomposition")
df_var = read_csv(var_alpha_path)
row = df_var.iloc[(df_var["alpha"] - 1.0).abs().argsort()[:1]].iloc[0]
factor_labels = ["Moisture\n(cM)", "Residue\n(cR)", "Tillage\n(cT)", "GDD\n(cG)"]
var_keys = ["Var_cM", "Var_cR", "Var_cT", "Var_cG"]
var_vals = np.array([float(row[k]) for k in var_keys])
var_pct = 100.0 * var_vals / var_vals.sum()

fig, ax = plt.subplots(figsize=(COL2, 3.2), constrained_layout=True)

y_pos = np.arange(len(factor_labels))
colors = SERIES[:4]

# Horizontal lollipop
for i, (yp, val, pct, col) in enumerate(zip(y_pos, var_vals, var_pct, colors)):
    ax.plot([0, val], [yp, yp], color=col, linewidth=2.5, solid_capstyle="round")
    ax.scatter(val, yp, color=col, s=70, zorder=5, edgecolors="white", linewidths=1.0)
    ax.text(val + var_vals.max() * 0.03, yp, f"{val:.4f}  ({pct:.1f}%)",
            va="center", fontsize=7, color=PAL["dgray"])

ax.set_yticks(y_pos)
ax.set_yticklabels(factor_labels)
ax.set_xlabel("Variance")
ax.set_title(r"Variance Decomposition ($\alpha$ = 1)")
ax.set_xlim(left=0)
_add_grid(ax, "x")
ax.invert_yaxis()

_save(fig, "F3_variance_decomposition")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Residue Damping (2-panel: corr + rank stability)    ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F4: Residue damping (2-panel)")
df_a = read_csv(alpha_sweep_path).sort_values("alpha")
alphas = df_a["alpha"].astype(float).to_numpy()
corr_col = get_col(df_a, ["corr_Teff_R"])
sp_col = get_col(df_a, ["spearman_vs_alpha1"])
corr_vals = df_a[corr_col].astype(float).to_numpy()
sp_vals = df_a[sp_col].astype(float).to_numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 3.0), constrained_layout=True)

# Panel (a): corr(T_eff, R)
ax1.fill_between(alphas, corr_vals, color=PAL["blue"], alpha=0.08)
ax1.plot(alphas, corr_vals, color=PAL["blue"], linewidth=1.8,
         marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"corr($T_{\mathrm{eff}}$, $R$)")
ax1.set_title("Factor Dependence")
_add_grid(ax1, "both")
_panel_label(ax1, "(a)")

# Panel (b): Spearman vs alpha=1
ax2.fill_between(alphas, sp_vals, color=PAL["orange"], alpha=0.08)
ax2.plot(alphas, sp_vals, color=PAL["orange"], linewidth=1.8,
         marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"Spearman $\rho$ vs $\alpha$=1")
ax2.set_title("Rank Stability")
_add_grid(ax2, "both")
_panel_label(ax2, "(b)")

_stat_annotation(ax2, [
    f"min = {sp_vals.min():.4f}",
    f"max = {sp_vals.max():.4f}",
], x=0.03, y=0.15, va="bottom")

_save(fig, "F4_residue_damping")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Geometric vs Additive Scatter (w/ marginals)        ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F5: Geometric vs additive scatter")
df2r = read_csv(step2_row_path)
x_geo = df2r[get_col(df2r, ["score_geo"])].astype(float).to_numpy()
y_add = df2r[get_col(df2r, ["score_add"])].astype(float).to_numpy()
rho = spearman_corr(x_geo, y_add)
disagreement = np.abs(y_add - x_geo)

# Create figure with marginal axes
fig = plt.figure(figsize=(COL15, COL15 * 0.95))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                      hspace=0.04, wspace=0.04)
ax_main = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

# Main scatter
sc = ax_main.scatter(x_geo, y_add, s=12, c=disagreement, cmap=DIV_CMAP,
                     alpha=0.7, edgecolors="none", rasterized=True)
lims = [min(x_geo.min(), y_add.min()) - 2, max(x_geo.max(), y_add.max()) + 2]
ax_main.plot(lims, lims, "--", color=PAL["lgray"], linewidth=1.0, zorder=0)
ax_main.set_xlim(lims)
ax_main.set_ylim(lims)
ax_main.set_xlabel("Geometric fusion score")
ax_main.set_ylabel("Additive fusion score")
ax_main.set_aspect("equal")
_add_grid(ax_main, "both")

_stat_annotation(ax_main, [f"Spearman = {rho:.4f}", f"n = {len(x_geo)}"],
                 x=0.03, y=0.97)

# Marginal histograms
ax_histx.hist(x_geo, bins=40, color=PAL["blue"], alpha=0.55, edgecolor="white", linewidth=0.3)
ax_histy.hist(y_add, bins=40, orientation="horizontal", color=PAL["orange"],
              alpha=0.55, edgecolor="white", linewidth=0.3)
ax_histx.set_title("Geometric vs Additive Fusion", fontsize=9, fontweight="bold")
ax_histx.tick_params(labelbottom=False)
ax_histy.tick_params(labelleft=False)
_despine(ax_histx)
_despine(ax_histy)
ax_histy.spines["bottom"].set_visible(False)
ax_histx.set_ylabel("Count", fontsize=6.5)
ax_histy.set_xlabel("Count", fontsize=6.5)

# Colorbar
cax = fig.add_axes([0.78, 0.12, 0.015, 0.25])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("|Disagreement|", fontsize=6.5)
cb.ax.tick_params(labelsize=6)

_save(fig, "F5_geo_vs_add_scatter")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — LOFO Rank Stability (dot + bar)                     ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F6: LOFO rank stability")
df4 = read_csv(step4_lofo_path)
factors = df4[get_col(df4, ["removed_factor"])].astype(str).tolist()
lofo_vals = df4[get_col(df4, ["spearman_vs_full"])].astype(float).to_numpy()

# Sort by impact (ascending spearman = more impact)
sort_idx = np.argsort(lofo_vals)
factors_s = [factors[i] for i in sort_idx]
lofo_s = lofo_vals[sort_idx]
colors_s = [SERIES[i % len(SERIES)] for i in sort_idx]

fig, ax = plt.subplots(figsize=(COL2, 2.8), constrained_layout=True)

y_pos = np.arange(len(factors_s))
ax.barh(y_pos, lofo_s, color=colors_s, edgecolor="white", linewidth=0.5, height=0.55, alpha=0.7)
ax.scatter(lofo_s, y_pos, color=colors_s, s=50, zorder=5, edgecolors="white", linewidths=0.8)

for yp, val in zip(y_pos, lofo_s):
    ax.text(val + 0.002, yp, f"{val:.4f}", va="center", fontsize=6.5, color=PAL["dgray"])

ax.set_yticks(y_pos)
ax.set_yticklabels(factors_s, fontsize=7)
ax.set_xlabel(r"Spearman $\rho$ vs full model")
ax.set_title("Leave-One-Factor-Out Rank Stability")
ax.set_xlim(left=min(0, lofo_s.min() - 0.02),
            right=min(1.02, lofo_s.max() + 0.04))
_add_grid(ax, "x")

# Annotate most/least impactful
ax.annotate("Most impactful", xy=(lofo_s[0], 0), fontsize=6,
            color=PAL["vermil"], ha="left", va="bottom",
            xytext=(lofo_s[0] - 0.01, -0.5))

_save(fig, "F6_lofo_stability")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7 — Weight Sweep Robustness (line + confidence band)    ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F7: Weight sweep robustness")
df3 = read_csv(step3_weights_path)
test_col = get_col(df3, ["test"])
spb_col = get_col(df3, ["spearman_vs_base"])
mask = df3[test_col].astype(str).str.startswith("sweep_")
df3s = df3[mask].copy().reset_index(drop=True)
sweep_x = np.arange(len(df3s))
sweep_y = df3s[spb_col].astype(float).to_numpy()

fig, ax = plt.subplots(figsize=(COL2, 3.2), constrained_layout=True)

# Confidence band around mean
mean_y = sweep_y.mean()
std_y = sweep_y.std()
ax.axhspan(mean_y - std_y, mean_y + std_y, color=PAL["green"], alpha=0.08)
ax.axhline(mean_y, color=PAL["green"], linewidth=0.8, linestyle=":", alpha=0.5)

ax.fill_between(sweep_x, sweep_y, mean_y, color=PAL["green"], alpha=0.12)
ax.plot(sweep_x, sweep_y, color=PAL["green"], linewidth=1.6,
        marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)

ax.set_xlabel("Weight configuration index")
ax.set_ylabel(r"Spearman $\rho$ vs baseline")
ax.set_title("Weight Sweep Robustness")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
_add_grid(ax, "y")

_stat_annotation(ax, [
    f"n = {len(sweep_y)} configs",
    f"mean = {mean_y:.4f}",
    f"std  = {std_y:.4f}",
    f"min  = {sweep_y.min():.4f}",
], x=0.97, y=0.03, ha="right", va="bottom")

_save(fig, "F7_weight_robustness")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 8 — Scaler Comparison (grouped bars, clean)             ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F8: Scaler comparison")
df5 = read_csv(step5_scaler_path)
scaler_col = get_col(df5, ["scaler"])
classes = ["A", "B", "C", "D", "E"]
labels = df5[scaler_col].astype(str).tolist()
n_scalers = len(labels)

x = np.arange(len(classes))
width = 0.72 / max(n_scalers, 1)

fig, ax = plt.subplots(figsize=(COL2, 3.6), constrained_layout=True)

for i, lab in enumerate(labels):
    vals = [float(df5.loc[df5[scaler_col] == lab, c].values[0]) for c in classes]
    offset = (i - (n_scalers - 1) / 2.0) * width
    ax.bar(x + offset, vals, width=width * 0.9, label=lab,
           color=SERIES[i % len(SERIES)], edgecolor="white", linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_xlabel("Risk class")
ax.set_ylabel("Image count")
ax.set_title("Scaling Method Comparison by Risk Class")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
_add_grid(ax, "y")

ax.legend(title="Scaler", ncols=min(n_scalers, 4),
          loc="upper center", bbox_to_anchor=(0.5, -0.15),
          frameon=True, edgecolor=PAL["vlgray"], fancybox=False)

_save(fig, "F8_scaler_comparison")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 9 — Noise Robustness (2-panel: Spearman + Jaccard)      ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F9: Noise robustness (2-panel)")
df6 = read_csv(step6_noise_path)
noise_x = df6[get_col(df6, ["noise_sigma"])].astype(float).to_numpy()
noise_s = df6[get_col(df6, ["spearman_vs_base"])].astype(float).to_numpy()
noise_j = df6[get_col(df6, ["top20_jaccard_vs_base"])].astype(float).to_numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 3.0), constrained_layout=True)

# Panel (a): Spearman
ax1.fill_between(noise_x, noise_s, color=PAL["blue"], alpha=0.08)
ax1.plot(noise_x, noise_s, color=PAL["blue"], linewidth=1.8,
         marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
# Highlight degradation threshold
thresh = 0.95
if noise_s.min() < thresh:
    ax1.axhline(thresh, color=PAL["vermil"], linewidth=0.7, linestyle="--", alpha=0.6)
    ax1.text(noise_x[-1], thresh + 0.005, f"$\\rho$ = {thresh}", fontsize=6,
             color=PAL["vermil"], ha="right", va="bottom")
ax1.set_xlabel(r"Noise $\sigma$")
ax1.set_ylabel(r"Spearman $\rho$ vs baseline")
ax1.set_title("Rank Stability")
_add_grid(ax1, "both")
_panel_label(ax1, "(a)")

# Panel (b): Jaccard
ax2.fill_between(noise_x, noise_j, color=PAL["orange"], alpha=0.08)
ax2.plot(noise_x, noise_j, color=PAL["orange"], linewidth=1.8,
         marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
if noise_j.min() < thresh:
    ax2.axhline(thresh, color=PAL["vermil"], linewidth=0.7, linestyle="--", alpha=0.6)
ax2.set_xlabel(r"Noise $\sigma$")
ax2.set_ylabel("Top-20% Jaccard vs baseline")
ax2.set_title("High-Risk Subset Stability")
_add_grid(ax2, "both")
_panel_label(ax2, "(b)")

_save(fig, "F9_noise_robustness")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 10 — Dirichlet Rank Uncertainty (histogram + KDE)       ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F10: Dirichlet rank uncertainty")
df7 = read_csv(step7_rank_stats_path)
rank_std   = df7[get_col(df7, ["rank_std"])].astype(float).to_numpy()
rank_mean  = df7[get_col(df7, ["rank_mean"])].astype(float).to_numpy()
top20_prob = df7[get_col(df7, ["top20_prob"])].astype(float).to_numpy()

fig, ax = plt.subplots(figsize=(COL2, 3.4), constrained_layout=True)

n_h, bins_h, patches_h = ax.hist(rank_std, bins=35, color=PAL["blue"],
                                  alpha=0.55, edgecolor="white", linewidth=0.4)

# KDE overlay
_kde_on_hist(ax, rank_std, PAL["vermil"], bw_method=0.3)

# Percentile markers
med_rs = np.median(rank_std)
p95 = np.percentile(rank_std, 95)
ymax = ax.get_ylim()[1]
ax.axvline(med_rs, color=PAL["dgray"], linewidth=0.8, linestyle="--", alpha=0.7)
ax.axvline(p95, color=PAL["vermil"], linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(med_rs, ymax * 0.95, f"  median={med_rs:.1f}", fontsize=6, color=PAL["dgray"], va="top")
ax.text(p95, ymax * 0.85, f"  95th={p95:.1f}", fontsize=6, color=PAL["vermil"], va="top")

ax.set_xlabel("Rank standard deviation")
ax.set_ylabel("Frequency")
ax.set_title("Dirichlet Ensemble Rank Uncertainty")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
_add_grid(ax, "y")

_save(fig, "F10_rank_uncertainty")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FIGURE 11 — Rank Stability vs Mean Rank (styled scatter)       ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("F11: Rank stability vs mean rank")
robust_mask = top20_prob >= 0.9

fig, ax = plt.subplots(figsize=(COL15, 4.0), constrained_layout=True)

# Background points
ax.scatter(rank_mean[~robust_mask], rank_std[~robust_mask],
           s=10, color=PAL["lgray"], alpha=0.5, edgecolors="none",
           rasterized=True, label="Others")
# Highlighted points
ax.scatter(rank_mean[robust_mask], rank_std[robust_mask],
           s=22, color=PAL["vermil"], alpha=0.8,
           edgecolors="white", linewidths=0.4,
           rasterized=True, label=r"$P_{top20}$ $\geq$ 0.9")

ax.set_xlabel("Mean rank (Dirichlet ensemble)")
ax.set_ylabel("Rank standard deviation")
ax.set_title("Rank Stability vs Mean Rank")
_add_grid(ax, "both")

ax.legend(loc="upper right", markerscale=1.5, framealpha=0.9,
          edgecolor=PAL["vlgray"])

_stat_annotation(ax, [
    f"n robust = {robust_mask.sum()} / {len(robust_mask)}",
    f"median rank_std = {np.median(rank_std):.1f}",
], x=0.03, y=0.97)

_save(fig, "F11_rank_vs_stability")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SUPPLEMENTARY X1 — Log-Space Contributions (violin + strip)    ║
# ╚═══════════════════════════════════════════════════════════════════╝
have_cols = all(c in df_risk.columns for c in ["M", "R", "T_eff", "G"])
if have_cols:
    print("X1: Contribution violins")
    M = df_risk["M"].astype(float).to_numpy()
    R = df_risk["R"].astype(float).to_numpy()
    U = 1 - R
    T = df_risk["T_eff"].astype(float).to_numpy()
    G = df_risk["G"].astype(float).to_numpy()
    cM = wM * np.log(np.maximum(M, eps))
    cR = wR * np.log(np.maximum(U, eps))
    cT = wT * np.log(np.maximum(T, eps))
    cG = wG * np.log(np.maximum(G, eps))

    data_list = [cM, cR, cT, cG]
    labels_v = [r"$c_M$", r"$c_R$", r"$c_T$", r"$c_G$"]

    fig, ax = plt.subplots(figsize=(COL2, 3.8), constrained_layout=True)

    parts = ax.violinplot(data_list, positions=range(4), showmedians=False,
                          showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SERIES[i])
        pc.set_alpha(0.35)
        pc.set_edgecolor(SERIES[i])
        pc.set_linewidth(0.8)

    # Strip (jittered points)
    for i, d in enumerate(data_list):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full_like(d, i) + jitter, d, s=4, color=SERIES[i],
                   alpha=0.25, edgecolors="none", rasterized=True)
        # Median marker
        med = np.median(d)
        ax.scatter(i, med, s=40, color="white", zorder=5, edgecolors=SERIES[i], linewidths=1.5)
        ax.text(i + 0.3, med, f"{med:.3f}", fontsize=6, va="center", color=PAL["dgray"])

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_v)
    ax.set_ylabel("Log-space contribution")
    ax.set_title("Factor Contribution Distributions (Baseline)")
    _add_grid(ax, "y")

    _save(fig, "X1_contribution_violins")
else:
    print("Skipping X1 (baseline lacks M/R/T_eff/G columns)")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SUPPLEMENTARY X2 — Uncertainty vs Top-20 Probability           ║
# ╚═══════════════════════════════════════════════════════════════════╝
print("X2: Uncertainty vs top-20 probability")
fig, ax = plt.subplots(figsize=(COL15, 4.0), constrained_layout=True)

sc = ax.scatter(top20_prob, rank_std, s=12, c=rank_mean, cmap="viridis_r",
                alpha=0.7, edgecolors="none", rasterized=True)
ax.set_xlabel(r"Top-20% inclusion probability ($P_{top20}$)")
ax.set_ylabel("Rank standard deviation")
ax.set_title("Rank Uncertainty vs High-Risk Inclusion Probability")
_add_grid(ax, "both")

cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=30, pad=0.02)
cb.set_label("Mean rank", fontsize=7)
cb.ax.tick_params(labelsize=6)

# Highlight decision boundary
ax.axvline(0.9, color=PAL["vermil"], linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(0.91, ax.get_ylim()[1] * 0.95, r"$P$ = 0.9", fontsize=6.5,
        color=PAL["vermil"], va="top")

_save(fig, "X2_uncertainty_vs_prob")


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  All figures saved to: {OUT}")
print("=" * 60)
