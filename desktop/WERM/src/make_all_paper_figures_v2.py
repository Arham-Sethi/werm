import os, glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# CONFIG (edit only ROOT if needed)
# =========================
ROOT = "/content/drive/MyDrive/WERM"
EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT, exist_ok=True)

# Baseline fusion settings (must match your paper)
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30
eps = 1e-6

# Publication-oriented visual system
COLORS = {
    "navy": "#16324F",
    "teal": "#247BA0",
    "green": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "slate": "#5B6571",
    "ink": "#1F2933",
    "grid": "#D9E2EC",
    "panel": "#F7F9FC",
}
CLASS_COLORS = {
    "A": "#2A9D8F",
    "B": "#66BD63",
    "C": "#E9C46A",
    "D": "#F4A261",
    "E": "#E76F51",
}
SERIES_COLORS = [
    COLORS["navy"],
    COLORS["teal"],
    COLORS["green"],
    COLORS["gold"],
    COLORS["orange"],
    COLORS["red"],
]


# =========================
# Helpers
# =========================
def setup_plot_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": COLORS["slate"],
        "ytick.color": COLORS["slate"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "dejavuserif",
    })


def make_figure(size=(7.2, 4.6)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    return fig, ax


def style_axes(ax, title, xlabel, ylabel, subtitle=None, grid_axis="y"):
    ax.set_title(title, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color=COLORS["slate"],
            va="bottom",
        )


def add_stat_box(ax, lines, loc="upper left"):
    loc_map = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = loc_map[loc]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )


def savefig(fig, fname):
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", path)


def find_one(patterns):
    """Return first match among patterns (glob), else None."""
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            hits.sort()
            return hits[0]
    return None


def find_in_exp(exp_name, patterns):
    """Search inside experiments/<exp_name>/ for patterns, return first hit."""
    base = os.path.join(EXP, exp_name)
    pats = [os.path.join(base, "**", p) for p in patterns]
    return find_one(pats)


def read_csv(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_col(df, candidates):
    """Pick first matching column name."""
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


setup_plot_style()


# =========================
# Locate critical inputs
# =========================

# Step 0 baseline (preferred)
risk_base_path = find_in_exp("00_freeze", [
    "risk_eps_1e-6.csv",
    "*risk*1e-6*.csv",
    "*risk*eps*.csv",
])

# Frozen per-image base factors (fallback if baseline is missing)
freeze_path = find_in_exp("00_freeze", [
    "fusion_results_FREEZE.csv",
    "*FREEZE*.csv",
])

# Step 1 (variance + alpha sweep)
var_alpha_path = find_in_exp("01_residue_damping", ["variance_alpha0_alpha1.csv"])
alpha_sweep_path = find_in_exp("01_residue_damping", ["alpha_sweep_summary.csv"])

# Step 2
step2_row_path = find_in_exp("02_fusion_baselines", ["step2_scores_geo_add.csv", "*geo*add*.csv"])
step2_summary_path = find_in_exp("02_fusion_baselines", ["*summary*.csv"])

# Step 3
step3_weights_path = find_in_exp("03_weight_robustness", ["step3_weight_robustness_summary.csv", "*robustness*summary*.csv"])

# Step 4
step4_lofo_path = find_in_exp("04_lofo_ablation", ["step4_lofo_summary.csv", "*lofo*summary*.csv"])

# Step 5
step5_scaler_path = find_in_exp("05_scaling_comparison", ["step5_scaler_summary.csv", "*scaler*summary*.csv"])

# Step 6
step6_noise_path = find_in_exp("06_noise_robustness", ["step6_noise_summary.csv", "*noise*summary*.csv"])

# Step 7
step7_rank_stats_path = find_in_exp("07_dirichlet_weights", ["step7_image_rank_stats.csv"])
step7_dir_summary_path = find_in_exp("07_dirichlet_weights", ["step7_dirichlet_summary.csv", "*dirichlet*summary*.csv"])

print("Using files:")
for p in [
    risk_base_path,
    freeze_path,
    var_alpha_path,
    alpha_sweep_path,
    step2_row_path,
    step3_weights_path,
    step4_lofo_path,
    step5_scaler_path,
    step6_noise_path,
    step7_rank_stats_path,
    step7_dir_summary_path,
]:
    print(" -", p)


# =========================
# Ensure baseline risk exists (compute if missing)
# =========================
df_risk = None

if risk_base_path is not None:
    df_risk = read_csv(risk_base_path)
else:
    if freeze_path is None:
        raise FileNotFoundError(
            "Could not find Step 0 baseline risk CSV (risk_eps_1e-6.csv) AND could not find fusion_results_FREEZE.csv to recompute it."
        )
    print("\nBaseline risk file not found. Recomputing baseline risk from fusion_results_FREEZE.csv ...")
    df_freeze = read_csv(freeze_path)

    # Your known column names in FREEZE
    id_col = get_col(df_freeze, ["image", "image_id", "img", "filename"])
    m_col = get_col(df_freeze, ["moisture", "M"])
    r_col = get_col(df_freeze, ["residue", "R"])
    t_col = get_col(df_freeze, ["tillage_eff", "T_eff", "T"])
    g_col = get_col(df_freeze, ["gdd_norm", "G"])

    M = df_freeze[m_col].astype(float).to_numpy()
    R = df_freeze[r_col].astype(float).to_numpy()
    U = 1.0 - R
    T = df_freeze[t_col].astype(float).to_numpy()
    G = df_freeze[g_col].astype(float).to_numpy()

    log_risk = (
        wM * np.log(np.maximum(M, eps))
        + wG * np.log(np.maximum(G, eps))
        + wR * np.log(np.maximum(U, eps))
        + wT * np.log(np.maximum(T, eps))
    )
    risk_raw = np.exp(log_risk)
    score = robust_scale(log_risk)
    cls = risk_class(score).astype(str)

    df_risk = pd.DataFrame({
        "image": df_freeze[id_col].values,
        "M": M,
        "R": R,
        "U": U,
        "T_eff": T,
        "G": G,
        "risk_raw": risk_raw,
        "log_risk": log_risk,
        "risk_score_0_100": score,
        "risk_class": cls,
    })

    # Save it where the other scripts expect it (Drive)
    out_tables = os.path.join(EXP, "00_freeze", "tables")
    os.makedirs(out_tables, exist_ok=True)
    risk_base_path = os.path.join(out_tables, "risk_eps_1e-6.csv")
    df_risk.to_csv(risk_base_path, index=False)
    print("Recomputed + saved baseline to:", risk_base_path)


# Resolve baseline columns
score_col = get_col(df_risk, ["risk_score_0_100", "risk_score", "score"])
class_col = get_col(df_risk, ["risk_class", "class"])
id_col = get_col(df_risk, ["image", "image_id"])


# =========================
# FIGURE 1 - Risk histogram
# =========================
risk_scores = df_risk[score_col].astype(float).to_numpy()
fig, ax = make_figure()
n, bins, patches = ax.hist(
    risk_scores,
    bins=20,
    edgecolor="white",
    linewidth=0.8,
)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
norm = mpl.colors.Normalize(vmin=bins.min(), vmax=bins.max())
for center, patch in zip(bin_centers, patches):
    patch.set_facecolor(mpl.cm.cividis(norm(center)))
style_axes(
    ax,
    "Baseline Risk Score Distribution",
    "Risk score (0-100)",
    "Image count",
    subtitle="Histogram of robustly scaled baseline scores across the study set.",
    grid_axis="y",
)
add_stat_box(
    ax,
    [
        f"n = {len(risk_scores)}",
        f"mean = {risk_scores.mean():.1f}",
        f"median = {np.median(risk_scores):.1f}",
    ],
)
savefig(fig, "F1_risk_hist.png")


# =========================
# FIGURE 2 - A-E distribution
# =========================
counts = df_risk[class_col].value_counts()
order = ["A", "B", "C", "D", "E"]
vals = [int(counts.get(k, 0)) for k in order]
fig, ax = make_figure(size=(7.0, 4.8))
bars = ax.bar(
    order,
    vals,
    color=[CLASS_COLORS[k] for k in order],
    edgecolor="white",
    linewidth=1.0,
)
style_axes(
    ax,
    "Baseline Risk Class Composition",
    "Risk class",
    "Image count",
    subtitle="Discrete class allocation induced by the 0-100 risk score bins.",
    grid_axis="y",
)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.015, str(val), ha="center", va="bottom", fontsize=9)
savefig(fig, "F2_class_distribution.png")


# =========================
# FIGURE 3 - Variance decomposition (alpha=1)
# =========================
df_var = read_csv(var_alpha_path)
row = df_var.iloc[(df_var["alpha"] - 1.0).abs().argsort()[:1]].iloc[0]
vars_map = {
    "Var(cM)": float(row["Var_cM"]),
    "Var(cR)": float(row["Var_cR"]),
    "Var(cT)": float(row["Var_cT"]),
    "Var(cG)": float(row["Var_cG"]),
}
fig, ax = make_figure()
labels = list(vars_map.keys())
values = list(vars_map.values())
bars = ax.bar(labels, values, color=SERIES_COLORS[:4], edgecolor="white", linewidth=1.0)
style_axes(
    ax,
    "Variance Decomposition at alpha = 1",
    "Contribution term",
    "Variance",
    subtitle="Per-factor variance in log-space risk contributions for the reference damping setting.",
    grid_axis="y",
)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
savefig(fig, "F3_variance_decomposition_alpha1.png")


# =========================
# FIGURE 4A/4B - Alpha sweep
# =========================
df_a = read_csv(alpha_sweep_path)
alpha_col = get_col(df_a, ["alpha"])
corr_col = get_col(df_a, ["corr_Teff_R"])
sp_col = get_col(df_a, ["spearman_vs_alpha1"])
df_a = df_a.sort_values(alpha_col)
alphas = df_a[alpha_col].astype(float).to_numpy()

fig, ax = make_figure()
ax.plot(alphas, df_a[corr_col].astype(float), marker="o", markersize=6, linewidth=2.2, color=COLORS["teal"])
ax.fill_between(alphas, df_a[corr_col].astype(float), color=COLORS["teal"], alpha=0.12)
style_axes(
    ax,
    "Residue Damping Sensitivity",
    r"$\alpha$",
    "corr(T_eff, R)",
    subtitle="Pairwise dependence between tillage effectiveness and residue as damping varies.",
    grid_axis="both",
)
savefig(fig, "F4A_corr_Teff_R_vs_alpha.png")

fig, ax = make_figure()
spearman_vals = df_a[sp_col].astype(float).to_numpy()
ax.plot(alphas, spearman_vals, marker="o", markersize=6, linewidth=2.2, color=COLORS["navy"])
ax.fill_between(alphas, spearman_vals, color=COLORS["navy"], alpha=0.10)
style_axes(
    ax,
    "Rank Stability Across alpha",
    r"$\alpha$",
    "Spearman vs alpha = 1",
    subtitle="Monotonic agreement of image ranking relative to the reference alpha setting.",
    grid_axis="both",
)
add_stat_box(ax, [f"min = {spearman_vals.min():.3f}", f"max = {spearman_vals.max():.3f}"], loc="lower left")
savefig(fig, "F4B_spearman_vs_alpha.png")


# =========================
# FIGURE 5 - Geo vs Add scatter
# =========================
df2r = read_csv(step2_row_path)
xcol = get_col(df2r, ["score_geo"])
ycol = get_col(df2r, ["score_add"])
x_geo = df2r[xcol].astype(float).to_numpy()
y_add = df2r[ycol].astype(float).to_numpy()
lims = [min(x_geo.min(), y_add.min()), max(x_geo.max(), y_add.max())]
rho_geo_add = spearman_corr(x_geo, y_add)
fig, ax = make_figure(size=(6.4, 6.0))
ax.scatter(
    x_geo,
    y_add,
    s=28,
    c=np.abs(y_add - x_geo),
    cmap="viridis",
    alpha=0.85,
    edgecolors="white",
    linewidths=0.35,
)
ax.plot(lims, lims, linestyle="--", linewidth=1.4, color=COLORS["red"])
ax.set_xlim(lims)
ax.set_ylim(lims)
style_axes(
    ax,
    "Geometric vs Additive Fusion Scores",
    "Geometric score",
    "Additive score",
    subtitle="Point color encodes absolute disagreement between the two fusion schemes.",
    grid_axis="both",
)
add_stat_box(ax, [f"Spearman = {rho_geo_add:.3f}"], loc="upper left")
savefig(fig, "F5_geo_vs_add_scatter.png")


# =========================
# FIGURE 6 - LOFO Spearman
# =========================
df4 = read_csv(step4_lofo_path)
xf = get_col(df4, ["removed_factor"])
yf = get_col(df4, ["spearman_vs_full"])
factors = df4[xf].astype(str).tolist()
lofo_vals = df4[yf].astype(float).to_numpy()
fig, ax = make_figure()
bars = ax.bar(factors, lofo_vals, color=SERIES_COLORS[:len(factors)], edgecolor="white", linewidth=1.0)
style_axes(
    ax,
    "Leave-One-Factor-Out Rank Stability",
    "Removed factor",
    "Spearman vs full model",
    subtitle="Higher values indicate that removing the factor has limited effect on the global ranking.",
    grid_axis="y",
)
ax.set_ylim(0, min(1.02, max(1.0, lofo_vals.max() + 0.05)))
for bar, val in zip(bars, lofo_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
savefig(fig, "F6_lofo_spearman.png")


# =========================
# FIGURE 7 - Weight sweep Spearman
# =========================
df3 = read_csv(step3_weights_path)
test_col = get_col(df3, ["test"])
spb_col = get_col(df3, ["spearman_vs_base"])
mask = df3[test_col].astype(str).str.startswith("sweep_")
df3s = df3[mask].copy().reset_index(drop=True)
sweep_x = np.arange(len(df3s))
sweep_y = df3s[spb_col].astype(float).to_numpy()
fig, ax = make_figure()
ax.plot(sweep_x, sweep_y, marker="o", markersize=5.5, linewidth=2.1, color=COLORS["green"])
ax.fill_between(sweep_x, sweep_y, color=COLORS["green"], alpha=0.12)
style_axes(
    ax,
    "Weight Sweep Robustness",
    "Sweep run index",
    "Spearman vs baseline",
    subtitle="Ranking agreement across alternative weight configurations relative to the paper baseline.",
    grid_axis="both",
)
add_stat_box(ax, [f"runs = {len(sweep_y)}", f"mean = {sweep_y.mean():.3f}", f"min = {sweep_y.min():.3f}"], loc="lower left")
savefig(fig, "F7_weight_sweep_spearman.png")


# =========================
# FIGURE 8 - Scaling comparison class counts
# =========================
df5 = read_csv(step5_scaler_path)
scaler_col = get_col(df5, ["scaler"])
classes = ["A", "B", "C", "D", "E"]
labels = df5[scaler_col].astype(str).tolist()
x = np.arange(len(classes))
width = min(0.78 / max(len(labels), 1), 0.22)
fig, ax = make_figure(size=(8.4, 4.9))
for i, lab in enumerate(labels):
    vals = [float(df5.loc[df5[scaler_col] == lab, c].values[0]) for c in classes]
    offset = (i - (len(labels) - 1) / 2.0) * width
    ax.bar(
        x + offset,
        vals,
        width=width,
        label=lab,
        color=SERIES_COLORS[i % len(SERIES_COLORS)],
        edgecolor="white",
        linewidth=0.9,
    )
ax.set_xticks(x, classes)
style_axes(
    ax,
    "Scaler Comparison by Risk Class",
    "Risk class",
    "Image count",
    subtitle="Grouped bars compare how alternative scaling schemes redistribute class frequencies.",
    grid_axis="y",
)
ax.legend(title="Scaler", ncols=min(len(labels), 3), loc="upper center", bbox_to_anchor=(0.5, 1.02))
savefig(fig, "F8_scaler_class_counts.png")


# =========================
# FIGURE 9A/9B - Noise robustness
# =========================
df6 = read_csv(step6_noise_path)
nx = get_col(df6, ["noise_sigma"])
ns = get_col(df6, ["spearman_vs_base"])
nj = get_col(df6, ["top20_jaccard_vs_base"])
noise_x = df6[nx].astype(float).to_numpy()
noise_s = df6[ns].astype(float).to_numpy()
noise_j = df6[nj].astype(float).to_numpy()

fig, ax = make_figure()
ax.plot(noise_x, noise_s, marker="o", markersize=6, linewidth=2.2, color=COLORS["orange"])
ax.fill_between(noise_x, noise_s, color=COLORS["orange"], alpha=0.13)
style_axes(
    ax,
    "Noise Robustness of Global Ranking",
    "Noise sigma",
    "Spearman vs baseline",
    subtitle="Rank agreement after perturbing inputs with increasing noise amplitude.",
    grid_axis="both",
)
savefig(fig, "F9A_noise_spearman.png")

fig, ax = make_figure()
ax.plot(noise_x, noise_j, marker="o", markersize=6, linewidth=2.2, color=COLORS["red"])
ax.fill_between(noise_x, noise_j, color=COLORS["red"], alpha=0.13)
style_axes(
    ax,
    "Noise Robustness of Top-20 Percent Selection",
    "Noise sigma",
    "Top-20 percent Jaccard vs baseline",
    subtitle="Overlap of the highest-risk subset as noise increases.",
    grid_axis="both",
)
savefig(fig, "F9B_noise_jaccard.png")


# =========================
# FIGURE 10/11 - Dirichlet uncertainty
# =========================
df7 = read_csv(step7_rank_stats_path)
rs_col = get_col(df7, ["rank_std"])
rm_col = get_col(df7, ["rank_mean"])
p_col = get_col(df7, ["top20_prob"])
rank_std = df7[rs_col].astype(float).to_numpy()
rank_mean = df7[rm_col].astype(float).to_numpy()
top20_prob = df7[p_col].astype(float).to_numpy()

fig, ax = make_figure()
n, bins, patches = ax.hist(rank_std, bins=30, edgecolor="white", linewidth=0.8)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
norm = mpl.colors.Normalize(vmin=bins.min(), vmax=bins.max())
for center, patch in zip(bin_centers, patches):
    patch.set_facecolor(mpl.cm.magma(norm(center)))
style_axes(
    ax,
    "Dirichlet Ensemble Rank Uncertainty",
    "Rank standard deviation",
    "Image count",
    subtitle="Spread of ranking variability induced by sampling alternative weight vectors.",
    grid_axis="y",
)
add_stat_box(ax, [f"median = {np.median(rank_std):.2f}", f"95th pct = {np.percentile(rank_std, 95):.2f}"])
savefig(fig, "F10_rank_std_hist.png")

robust = top20_prob >= 0.9
fig, ax = make_figure(size=(6.8, 5.8))
ax.scatter(
    rank_mean[~robust],
    rank_std[~robust],
    s=26,
    color=COLORS["slate"],
    alpha=0.45,
    label="Others",
    edgecolors="none",
)
ax.scatter(
    rank_mean[robust],
    rank_std[robust],
    s=38,
    color=COLORS["red"],
    alpha=0.9,
    label="top20_prob >= 0.9",
    edgecolors="white",
    linewidths=0.35,
)
style_axes(
    ax,
    "Rank Stability vs Mean Rank",
    "Rank mean",
    "Rank standard deviation",
    subtitle="Robust high-risk candidates remain concentrated at high top-20 inclusion probability.",
    grid_axis="both",
)
ax.legend(loc="upper right")
savefig(fig, "F11_rankmean_vs_rankstd.png")


# =========================
# Optional extras
# =========================

# X1: Contribution boxplots (if we have M/R/T_eff/G in baseline)
have_cols = all(c in df_risk.columns for c in ["M", "R", "T_eff", "G"])
if have_cols:
    M = df_risk["M"].astype(float).to_numpy()
    R = df_risk["R"].astype(float).to_numpy()
    U = 1 - R
    T = df_risk["T_eff"].astype(float).to_numpy()
    G = df_risk["G"].astype(float).to_numpy()

    cM = wM * np.log(np.maximum(M, eps))
    cR = wR * np.log(np.maximum(U, eps))
    cT = wT * np.log(np.maximum(T, eps))
    cG = wG * np.log(np.maximum(G, eps))

    fig, ax = make_figure(size=(7.2, 5.0))
    box = ax.boxplot(
        [cM, cR, cT, cG],
        labels=["cM", "cR", "cT", "cG"],
        patch_artist=True,
        medianprops={"color": COLORS["ink"], "linewidth": 1.5},
        boxprops={"edgecolor": COLORS["ink"], "linewidth": 1.0},
        whiskerprops={"color": COLORS["slate"], "linewidth": 1.0},
        capprops={"color": COLORS["slate"], "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], SERIES_COLORS[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    style_axes(
        ax,
        "Log-Space Contribution Distributions",
        "Contribution term",
        "Contribution value",
        subtitle="Distribution of factor-wise log contributions under the baseline fusion weights.",
        grid_axis="y",
    )
    savefig(fig, "X1_contribution_boxplots.png")
else:
    print("Skipping X1_contribution_boxplots.png (baseline CSV lacks M/R/T_eff/G columns)")


# X2: top20_prob vs rank_std
fig, ax = make_figure(size=(6.8, 5.5))
ax.scatter(
    top20_prob,
    rank_std,
    s=32,
    c=rank_mean,
    cmap="cividis_r",
    alpha=0.8,
    edgecolors="white",
    linewidths=0.3,
)
style_axes(
    ax,
    "Uncertainty vs Top-20 Inclusion Probability",
    "Top-20 probability (Dirichlet)",
    "Rank standard deviation",
    subtitle="Color encodes mean rank, linking uncertainty to expected rank position.",
    grid_axis="both",
)
savefig(fig, "X2_prob_vs_rankstd.png")

print("\nDONE. Figures saved to:", OUT)
