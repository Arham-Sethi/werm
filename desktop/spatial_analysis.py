"""
WERM Spatial Diagnostics Pipeline
Parts 1-7: merge, semivariograms, Moran's I, block CV, risk map
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from pathlib import Path
import math

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:/Users/Arham/OneDrive/Desktop/school/advanced-calculator")
GRID_CSV = BASE / "grid_imgs.csv"
WERM_CSV = BASE / "spatial_werm_outputs.csv"
OUT_DIR  = BASE / "results" / "spatial_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: MERGE AND COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: MERGE AND COORDINATE CONVERSION")
print("=" * 70)

grid = pd.read_csv(GRID_CSV)
werm = pd.read_csv(WERM_CSV)

# Harmonize join key: grid has 'img_name', werm has 'image_id'
grid = grid.rename(columns={"img_name": "image_id"})

df = pd.merge(grid, werm, on="image_id", how="inner")
print(f"Grid rows: {len(grid)}, WERM rows: {len(werm)}, Merged: {len(df)}")

# Coordinate conversion: lat/lon -> meters relative to centroid
lat0 = df["lat"].mean()
lon0 = df["lon"].mean()
df["x_m"] = (df["lon"] - lon0) * 111320 * math.cos(math.radians(lat0))
df["y_m"] = (df["lat"] - lat0) * 110540

print(f"Centroid: lat={lat0:.6f}, lon={lon0:.6f}")
print(f"x_m range: {df['x_m'].min():.2f} to {df['x_m'].max():.2f} m")
print(f"y_m range: {df['y_m'].min():.2f} to {df['y_m'].max():.2f} m")

field_width  = df["x_m"].max() - df["x_m"].min()
field_height = df["y_m"].max() - df["y_m"].min()
field_diag   = math.sqrt(field_width**2 + field_height**2)
print(f"Field extent: {field_width:.1f} m x {field_height:.1f} m, diagonal={field_diag:.1f} m")

out_csv = OUT_DIR / "spatial_full.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: SANITY CHECK - GRID LAYOUT PLOT
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 2: SANITY CHECK GRID PLOT")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(10, 7))
sc = ax2.scatter(df["x_m"], df["y_m"], c=df["risk_score"],
                 cmap="viridis", s=80, edgecolors="black", linewidths=0.5,
                 vmin=0, vmax=100)
for _, r in df.iterrows():
    ax2.annotate(r["image_id"].replace("img", ""), (r["x_m"], r["y_m"]),
                 fontsize=5, ha="center", va="bottom", xytext=(0, 4),
                 textcoords="offset points")
plt.colorbar(sc, ax=ax2, label="Risk Score (0-100)")
ax2.set_xlabel("x (meters)")
ax2.set_ylabel("y (meters)")
ax2.set_title("Field Grid Layout - Points colored by WERM Risk Score")
ax2.set_aspect("equal")
fig2.tight_layout()
fig2.savefig(OUT_DIR / "fig_grid_layout.png", dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {OUT_DIR / 'fig_grid_layout.png'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: SEMIVARIOGRAMS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 3: SEMIVARIOGRAMS")
print("=" * 70)

coords = df[["x_m", "y_m"]].values
dist_matrix = squareform(pdist(coords))

N_LAGS = 15
MAX_LAG = field_diag / 2.0
lag_edges = np.linspace(0, MAX_LAG, N_LAGS + 1)
lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2.0
lag_width = lag_edges[1] - lag_edges[0]

VARS = ["moisture_norm", "residue_norm", "texture_norm", "risk_score"]
VAR_LABELS = ["Moisture", "Residue", "Texture", "Risk Score"]


def empirical_semivariogram(values, dist_mat, lag_edges):
    """Compute empirical semivariance for each lag bin."""
    n = len(values)
    gamma = []
    counts = []
    for k in range(len(lag_edges) - 1):
        lo, hi = lag_edges[k], lag_edges[k + 1]
        mask = (dist_mat > lo) & (dist_mat <= hi)
        # Upper triangle only to avoid double-counting
        mask_upper = np.triu(mask, k=1)
        idx = np.where(mask_upper)
        if len(idx[0]) < 2:
            gamma.append(np.nan)
            counts.append(0)
            continue
        diffs = values[idx[0]] - values[idx[1]]
        gamma.append(0.5 * np.mean(diffs ** 2))
        counts.append(len(idx[0]))
    return np.array(gamma), np.array(counts)


def spherical_model(h, nugget, sill, range_a):
    """Spherical variogram model."""
    h = np.asarray(h, dtype=float)
    result = np.where(
        h <= range_a,
        nugget + (sill - nugget) * (1.5 * h / range_a - 0.5 * (h / range_a) ** 3),
        sill
    )
    result = np.where(h == 0, 0.0, result)
    return result


# Compute empirical semivariograms and fit spherical models
semivario_results = {}

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
axes3 = axes3.ravel()

for i, (var, label) in enumerate(zip(VARS, VAR_LABELS)):
    vals = df[var].values
    gamma, counts = empirical_semivariogram(vals, dist_matrix, lag_edges)

    # Filter out NaN lags for fitting
    valid = ~np.isnan(gamma) & (counts >= 2)
    h_fit = lag_centers[valid]
    g_fit = gamma[valid]

    # Initial guesses
    nugget0 = max(g_fit[0] * 0.1, 1e-8) if len(g_fit) > 0 else 1e-4
    sill0   = np.max(g_fit) if len(g_fit) > 0 else np.var(vals)
    range0  = MAX_LAG * 0.4

    try:
        popt, _ = curve_fit(
            spherical_model, h_fit, g_fit,
            p0=[nugget0, sill0, range0],
            bounds=([0, 0, 1], [sill0 * 2, sill0 * 5, MAX_LAG]),
            maxfev=10000
        )
        nugget_fit, sill_fit, range_fit = popt
    except Exception as e:
        print(f"  WARNING: Spherical fit failed for {var}: {e}")
        nugget_fit, sill_fit, range_fit = nugget0, sill0, range0

    semivario_results[var] = {
        "nugget": nugget_fit,
        "sill": sill_fit,
        "range_m": range_fit,
        "label": label
    }

    # Plot
    ax = axes3[i]
    ax.scatter(lag_centers[valid], g_fit, c="steelblue", s=40, zorder=3,
               label="Empirical")
    h_smooth = np.linspace(0, MAX_LAG, 200)
    ax.plot(h_smooth, spherical_model(h_smooth, nugget_fit, sill_fit, range_fit),
            "r-", lw=2, label="Spherical fit")
    ax.axhline(sill_fit, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"Sill={sill_fit:.4f}")
    ax.axvline(range_fit, color="orange", ls="--", lw=0.8, alpha=0.7, label=f"Range={range_fit:.1f}m")
    ax.set_xlabel("Lag distance (m)")
    ax.set_ylabel("Semivariance")
    ax.set_title(f"{label}\nnugget={nugget_fit:.4f}, sill={sill_fit:.4f}, range={range_fit:.1f}m")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, MAX_LAG)

    print(f"  {label:15s}  nugget={nugget_fit:.6f}  sill={sill_fit:.6f}  range={range_fit:.2f} m")

fig3.suptitle("Empirical Semivariograms with Spherical Model Fits", fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "fig_semivariograms.png", dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {OUT_DIR / 'fig_semivariograms.png'}")

# Summary table
sv_table = pd.DataFrame([
    {"Variable": semivario_results[v]["label"],
     "Nugget": f'{semivario_results[v]["nugget"]:.6f}',
     "Sill": f'{semivario_results[v]["sill"]:.6f}',
     "Range (m)": f'{semivario_results[v]["range_m"]:.2f}'}
    for v in VARS
])
sv_table.to_csv(OUT_DIR / "semivariogram_params.csv", index=False)
print(f"Saved: {OUT_DIR / 'semivariogram_params.csv'}\n")

median_range = float(np.median([semivario_results[v]["range_m"] for v in VARS]))
print(f"Median semivariogram range: {median_range:.2f} m\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: MORAN'S I
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 4: MORAN'S I")
print("=" * 70)

THRESHOLD = median_range


def morans_i(values, dist_mat, threshold):
    """Compute Moran's I with distance-band weights and permutation inference."""
    n = len(values)
    z = values - values.mean()

    # Binary distance-band weights
    W = (dist_mat <= threshold).astype(float)
    np.fill_diagonal(W, 0.0)

    W_sum = W.sum()
    if W_sum == 0:
        return {"I": np.nan, "E_I": np.nan, "z_score": np.nan, "p_value": np.nan}

    # Moran's I
    numerator = n * float(z @ W @ z)
    denominator = W_sum * float(z @ z)
    I = numerator / denominator if denominator != 0 else np.nan

    # Expected value
    E_I = -1.0 / (n - 1)

    # Permutation-based inference (999 permutations)
    n_perm = 999
    rng = np.random.default_rng(42)
    I_perm = np.empty(n_perm)
    for p in range(n_perm):
        z_perm = rng.permutation(z)
        num_p = n * float(z_perm @ W @ z_perm)
        den_p = W_sum * float(z_perm @ z_perm)
        I_perm[p] = num_p / den_p if den_p != 0 else np.nan

    I_mean = np.nanmean(I_perm)
    I_std  = np.nanstd(I_perm)
    z_score = (I - I_mean) / I_std if I_std > 0 else 0.0

    # Pseudo p-value (two-sided)
    p_value = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (n_perm + 1)

    return {"I": I, "E_I": E_I, "z_score": z_score, "p_value": p_value}


print(f"Distance-band threshold: {THRESHOLD:.2f} m (median semivariogram range)")

moran_results = {}
for var, label in zip(VARS, VAR_LABELS):
    vals = df[var].values
    res = morans_i(vals, dist_matrix, THRESHOLD)
    moran_results[var] = res
    sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
    print(f"  {label:15s}  I={res['I']:.4f}  E(I)={res['E_I']:.4f}  "
          f"z={res['z_score']:.3f}  p={res['p_value']:.4f}  {sig}")
    moran_results[var]["sig"] = sig

# Moran's I bar chart
fig4, ax4 = plt.subplots(figsize=(8, 5))
I_vals = [moran_results[v]["I"] for v in VARS]
colors_m = ["#2196F3" if moran_results[v]["p_value"] < 0.05 else "#BDBDBD" for v in VARS]
bars = ax4.bar(VAR_LABELS, I_vals, color=colors_m, edgecolor="black", width=0.5)

for bar, v in zip(bars, VARS):
    sig = moran_results[v]["sig"]
    y = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, y + 0.01,
             sig, ha="center", va="bottom", fontsize=14, fontweight="bold")

ax4.axhline(moran_results[VARS[0]]["E_I"], color="red", ls="--", lw=1,
            label=f'E(I) = {moran_results[VARS[0]]["E_I"]:.4f}')
ax4.set_ylabel("Moran's I")
ax4.set_title(f"Moran's I (distance-band = {THRESHOLD:.1f} m)\nBlue = significant (p<0.05)")
ax4.legend()
fig4.tight_layout()
fig4.savefig(OUT_DIR / "fig_morans_i.png", dpi=DPI, bbox_inches="tight")
plt.close(fig4)

moran_df = pd.DataFrame([
    {"Variable": VAR_LABELS[i], "Morans_I": f'{moran_results[v]["I"]:.4f}',
     "Expected_I": f'{moran_results[v]["E_I"]:.4f}',
     "z_score": f'{moran_results[v]["z_score"]:.3f}',
     "p_value": f'{moran_results[v]["p_value"]:.4f}',
     "Significance": moran_results[v]["sig"]}
    for i, v in enumerate(VARS)
])
moran_df.to_csv(OUT_DIR / "morans_i_results.csv", index=False)
print(f"Saved: {OUT_DIR / 'fig_morans_i.png'}")
print(f"Saved: {OUT_DIR / 'morans_i_results.csv'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 5: SPATIAL BLOCK CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 5: SPATIAL BLOCK CROSS-VALIDATION")
print("=" * 70)

BLOCK_SIZE = median_range
x_min, x_max = df["x_m"].min(), df["x_m"].max()
y_min, y_max = df["y_m"].min(), df["y_m"].max()

# Assign each point to a block
df["block_col"] = ((df["x_m"] - x_min) / BLOCK_SIZE).astype(int)
df["block_row"] = ((df["y_m"] - y_min) / BLOCK_SIZE).astype(int)
df["block_id"] = df["block_row"].astype(str) + "_" + df["block_col"].astype(str)

# Drop blocks with < 3 points
block_counts = df["block_id"].value_counts()
valid_blocks = block_counts[block_counts >= 3].index
df_blocks = df[df["block_id"].isin(valid_blocks)].copy()

print(f"Block size: {BLOCK_SIZE:.2f} m")
print(f"Total blocks: {df['block_id'].nunique()}")
print(f"Blocks with >= 3 points: {len(valid_blocks)}")
print(f"Points in valid blocks: {len(df_blocks)} / {len(df)}")

# Per-block stats
block_stats = df_blocks.groupby("block_id")["risk_score"].agg(["mean", "std", "count"])
block_stats["cv"] = block_stats["std"] / block_stats["mean"]
block_stats = block_stats.replace([np.inf, -np.inf], np.nan).dropna()

# Variance decomposition
total_var = df_blocks["risk_score"].var()
between_var = block_stats["mean"].var()
var_ratio = between_var / total_var if total_var > 0 else 0
mean_cv = block_stats["cv"].mean()

print(f"Between-block variance / total variance: {var_ratio:.4f}")
print(f"Mean within-block CV: {mean_cv:.4f}")

# Block map
fig5, ax5 = plt.subplots(figsize=(11, 8))

# Draw block grid
n_cols_grid = int((x_max - x_min) / BLOCK_SIZE) + 2
n_rows_grid = int((y_max - y_min) / BLOCK_SIZE) + 2

for bi in range(n_rows_grid):
    for bj in range(n_cols_grid):
        bx = x_min + bj * BLOCK_SIZE
        by = y_min + bi * BLOCK_SIZE
        rect = Rectangle((bx, by), BLOCK_SIZE, BLOCK_SIZE,
                          linewidth=1, edgecolor="gray", facecolor="none",
                          linestyle="--", alpha=0.5)
        ax5.add_patch(rect)

# Plot points
sc5 = ax5.scatter(df["x_m"], df["y_m"], c=df["risk_score"],
                  cmap="viridis", s=60, edgecolors="black", linewidths=0.5,
                  vmin=0, vmax=100, zorder=3)
plt.colorbar(sc5, ax=ax5, label="Risk Score")

# Label blocks with mean risk
for bid, stats in block_stats.iterrows():
    block_pts = df_blocks[df_blocks["block_id"] == bid]
    cx = block_pts["x_m"].mean()
    cy = block_pts["y_m"].mean()
    # Position label at block center
    bc = int(bid.split("_")[1])
    br = int(bid.split("_")[0])
    bx_center = x_min + bc * BLOCK_SIZE + BLOCK_SIZE / 2
    by_center = y_min + br * BLOCK_SIZE + BLOCK_SIZE / 2
    ax5.text(bx_center, by_center, f'{stats["mean"]:.0f}',
             ha="center", va="center", fontsize=9, fontweight="bold",
             color="red", bbox=dict(boxstyle="round,pad=0.15",
                                     facecolor="white", edgecolor="red",
                                     alpha=0.7), zorder=4)

ax5.set_xlabel("x (meters)")
ax5.set_ylabel("y (meters)")
ax5.set_title(f"Spatial Block Cross-Validation\n"
              f"Block size = {BLOCK_SIZE:.1f}m | "
              f"Between/Total var = {var_ratio:.3f} | "
              f"Mean CV = {mean_cv:.3f}")
ax5.set_aspect("equal")
fig5.tight_layout()
fig5.savefig(OUT_DIR / "fig_spatial_blocks.png", dpi=DPI, bbox_inches="tight")
plt.close(fig5)

block_stats.to_csv(OUT_DIR / "block_stats.csv")
print(f"Saved: {OUT_DIR / 'fig_spatial_blocks.png'}")
print(f"Saved: {OUT_DIR / 'block_stats.csv'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 6: INTERPOLATED RISK MAP
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 6: INTERPOLATED RISK MAP")
print("=" * 70)

# Create fine grid
margin = 2.0
xi = np.linspace(x_min - margin, x_max + margin, 300)
yi = np.linspace(y_min - margin, y_max + margin, 300)
XI, YI = np.meshgrid(xi, yi)

# Interpolate with cubic (fallback to linear if degenerate)
try:
    ZI = griddata(coords, df["risk_score"].values, (XI, YI), method="cubic")
except Exception:
    ZI = griddata(coords, df["risk_score"].values, (XI, YI), method="linear")

# Also do nearest-neighbor for the mask (to avoid extrapolation artifacts)
ZI_nn = griddata(coords, df["risk_score"].values, (XI, YI), method="nearest")

# Mask far-from-data regions using convex hull distance
from scipy.spatial import ConvexHull, Delaunay

try:
    hull = Delaunay(coords)
    mask_inside = hull.find_simplex(np.column_stack([XI.ravel(), YI.ravel()])) >= 0
    mask_inside = mask_inside.reshape(XI.shape)
    # Small buffer: also include points within ~2m of hull
    ZI_plot = np.where(mask_inside, ZI, np.nan)
    # Fill NaN from cubic with nearest neighbor inside hull
    nan_mask = np.isnan(ZI_plot) & mask_inside
    ZI_plot[nan_mask] = ZI_nn[nan_mask]
except Exception:
    ZI_plot = ZI

fig6, ax6 = plt.subplots(figsize=(11, 8))

# Filled contour
levels = np.linspace(0, 100, 21)
cf = ax6.contourf(XI, YI, ZI_plot, levels=levels, cmap="RdYlGn_r",
                  extend="both", alpha=0.9)
# Contour lines
ax6.contour(XI, YI, ZI_plot, levels=levels[::2], colors="black",
            linewidths=0.3, alpha=0.4)

# Overlay sampling points
ax6.scatter(df["x_m"], df["y_m"], c="black", s=20, zorder=5,
            edgecolors="white", linewidths=0.3)

cbar = plt.colorbar(cf, ax=ax6, label="WERM Risk Score (0-100)", shrink=0.85)
cbar.set_ticks(np.arange(0, 101, 10))

ax6.set_xlabel("x (meters)")
ax6.set_ylabel("y (meters)")
ax6.set_title("WERM Spatial Risk Map - Interpolated Surface\n"
              "(cubic interpolation, RdYlGn_r colormap: red=high risk, green=low)")
ax6.set_aspect("equal")
fig6.tight_layout()
fig6.savefig(OUT_DIR / "fig_risk_map.png", dpi=DPI, bbox_inches="tight")
plt.close(fig6)
print(f"Saved: {OUT_DIR / 'fig_risk_map.png'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 7: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 7: COMPREHENSIVE SUMMARY")
print("=" * 70)

print(f"\n  Points analyzed:          {len(df)}")
print(f"  Field extent:             {field_width:.1f} m x {field_height:.1f} m")
print(f"  Field diagonal:           {field_diag:.1f} m")
print(f"  Grid rows:                {df['row'].nunique()} | cols: {df['col'].nunique()}")

print(f"\n  --- Semivariogram Ranges (spatial autocorrelation scale) ---")
for v, label in zip(VARS, VAR_LABELS):
    r = semivario_results[v]
    print(f"  {label:15s}  range = {r['range_m']:6.2f} m  "
          f"(nugget={r['nugget']:.4f}, sill={r['sill']:.4f})")
print(f"  Median range:             {median_range:.2f} m")

print(f"\n  --- Moran's I (spatial clustering) ---")
for v, label in zip(VARS, VAR_LABELS):
    r = moran_results[v]
    interp = "CLUSTERED" if r["I"] > 0 and r["p_value"] < 0.05 else \
             "DISPERSED" if r["I"] < 0 and r["p_value"] < 0.05 else "RANDOM"
    print(f"  {label:15s}  I={r['I']:.4f}  p={r['p_value']:.4f}  -> {interp} {r['sig']}")

print(f"\n  --- Block Cross-Validation ---")
print(f"  Block size:               {BLOCK_SIZE:.1f} m")
print(f"  Valid blocks (>=3 pts):   {len(valid_blocks)}")
print(f"  Between/total var ratio:  {var_ratio:.4f}")
print(f"  Mean within-block CV:     {mean_cv:.4f}")

print(f"\n  --- Risk Score Distribution ---")
print(f"  Mean:   {df['risk_score'].mean():.2f}")
print(f"  Median: {df['risk_score'].median():.2f}")
print(f"  Std:    {df['risk_score'].std():.2f}")
print(f"  Min:    {df['risk_score'].min():.2f}")
print(f"  Max:    {df['risk_score'].max():.2f}")

print(f"\n  --- Interpretation for Paper ---")

any_clustered = any(moran_results[v]["I"] > 0 and moran_results[v]["p_value"] < 0.05 for v in VARS)
if any_clustered:
    clustered_vars = [VAR_LABELS[i] for i, v in enumerate(VARS)
                      if moran_results[v]["I"] > 0 and moran_results[v]["p_value"] < 0.05]
    print(f"  Significant spatial clustering detected in: {', '.join(clustered_vars)}")
    print(f"  This confirms WERM risk scores capture genuine spatial structure")
    print(f"  in the field, not random noise.")
else:
    print(f"  No significant spatial clustering detected. Risk variation")
    print(f"  may be driven by fine-grained or non-spatial factors.")

if var_ratio > 0.3:
    print(f"  High between-block variance ({var_ratio:.1%}) indicates distinct")
    print(f"  risk zones exist within the field at the {BLOCK_SIZE:.0f}m scale.")
elif var_ratio > 0.1:
    print(f"  Moderate between-block variance ({var_ratio:.1%}) suggests")
    print(f"  some spatial zoning of risk at the {BLOCK_SIZE:.0f}m scale.")
else:
    print(f"  Low between-block variance ({var_ratio:.1%}); risk is relatively")
    print(f"  homogeneous at the {BLOCK_SIZE:.0f}m scale.")

print(f"\n  Semivariogram ranges ({median_range:.0f}m median) indicate that spatial")
print(f"  autocorrelation persists over distances comparable to")
print(f"  {median_range / field_diag * 100:.0f}% of the field diagonal.")

# Save summary to text file
summary_path = OUT_DIR / "analysis_summary.txt"
with open(summary_path, "w") as f:
    f.write("WERM Spatial Diagnostics Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Points: {len(df)}\n")
    f.write(f"Field: {field_width:.1f}m x {field_height:.1f}m (diag={field_diag:.1f}m)\n\n")
    f.write("Semivariogram Parameters:\n")
    for v, label in zip(VARS, VAR_LABELS):
        r = semivario_results[v]
        f.write(f"  {label}: nugget={r['nugget']:.6f}, sill={r['sill']:.6f}, range={r['range_m']:.2f}m\n")
    f.write(f"\nMedian range: {median_range:.2f}m\n\n")
    f.write("Moran's I:\n")
    for v, label in zip(VARS, VAR_LABELS):
        r = moran_results[v]
        f.write(f"  {label}: I={r['I']:.4f}, z={r['z_score']:.3f}, p={r['p_value']:.4f} {r['sig']}\n")
    f.write(f"\nBlock CV: blocks={len(valid_blocks)}, between/total={var_ratio:.4f}, mean_cv={mean_cv:.4f}\n")
    f.write(f"Risk: mean={df['risk_score'].mean():.2f}, std={df['risk_score'].std():.2f}\n")

print(f"\n  Saved: {summary_path}")
print(f"\n  All outputs in: {OUT_DIR}")
print("=" * 70)
print("DONE")
print("=" * 70)
