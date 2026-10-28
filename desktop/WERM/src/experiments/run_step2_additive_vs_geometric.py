import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/02_fusion_baselines"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------
# Columns (edit only if needed)
# ----------------------
ID_COL = "image"
M_COL = "moisture"
R_COL = "residue"
G_COL = "gdd_norm"
T_COL = "tillage_eff"   # alpha=1 already assumed

# ----------------------
# Weights
# ----------------------
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30
eps = 1e-6

# ----------------------
# Helpers
# ----------------------
def spearman_corr(a, b):
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ar, br)[0,1]

def robust_scale(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = 1e-12
    z = (x - med) / mad
    y = np.tanh(z)
    return 50 * (y + 1)

def risk_class(score):
    bins = [-1e9,20,40,60,80,1e9]
    labels = ["A","B","C","D","E"]
    return pd.cut(score, bins=bins, labels=labels, right=False)

def entropy(x, bins=10):
    hist, _ = np.histogram(x, bins=bins)
    p = hist / np.sum(hist)
    p = p[p>0]
    return -np.sum(p * np.log(p))

# ----------------------
# Load
# ----------------------
df = pd.read_csv(INPUT_CSV)

M = df[M_COL].values
R = df[R_COL].values
U = 1 - R
G = df[G_COL].values
T = df[T_COL].values

# ----------------------
# Geometric model
# ----------------------
Mc = np.maximum(M, eps)
Uc = np.maximum(U, eps)
Gc = np.maximum(G, eps)
Tc = np.maximum(T, eps)

log_geo = (
    wM*np.log(Mc) +
    wG*np.log(Gc) +
    wR*np.log(Uc) +
    wT*np.log(Tc)
)

score_geo = robust_scale(log_geo)
class_geo = risk_class(score_geo)

# ----------------------
# Additive model
# ----------------------
risk_add_raw = (
    wM*M +
    wG*G +
    wR*U +
    wT*T
)

score_add = robust_scale(risk_add_raw)
class_add = risk_class(score_add)

# ----------------------
# Metrics
# ----------------------
spear = spearman_corr(score_geo, score_add)

top_k = int(0.2 * len(score_geo))
geo_top = set(np.argsort(score_geo)[-top_k:])
add_top = set(np.argsort(score_add)[-top_k:])
jaccard = len(geo_top & add_top) / len(geo_top | add_top)

entropy_geo = entropy(score_geo)
entropy_add = entropy(score_add)

counts_geo = class_geo.value_counts()
counts_add = class_add.value_counts()

corr_geo = {
    "corr_geo_M": np.corrcoef(score_geo, M)[0,1],
    "corr_geo_U": np.corrcoef(score_geo, U)[0,1],
    "corr_geo_T": np.corrcoef(score_geo, T)[0,1],
    "corr_geo_G": np.corrcoef(score_geo, G)[0,1],
}

corr_add = {
    "corr_add_M": np.corrcoef(score_add, M)[0,1],
    "corr_add_U": np.corrcoef(score_add, U)[0,1],
    "corr_add_T": np.corrcoef(score_add, T)[0,1],
    "corr_add_G": np.corrcoef(score_add, G)[0,1],
}

# ----------------------
# Save row-level file
# ----------------------
row_df = pd.DataFrame({
    ID_COL: df[ID_COL],
    "score_geo": score_geo,
    "class_geo": class_geo,
    "score_add": score_add,
    "class_add": class_add
})

row_df.to_csv(os.path.join(TABLE_DIR, "step2_scores_geo_add.csv"), index=False)

# ----------------------
# Save summary
# ----------------------
summary = {
    "spearman_geo_add": spear,
    "top20_jaccard": jaccard,
    "entropy_geo": entropy_geo,
    "entropy_add": entropy_add,
    "A_geo": counts_geo.get("A",0),
    "B_geo": counts_geo.get("B",0),
    "C_geo": counts_geo.get("C",0),
    "D_geo": counts_geo.get("D",0),
    "E_geo": counts_geo.get("E",0),
    "A_add": counts_add.get("A",0),
    "B_add": counts_add.get("B",0),
    "C_add": counts_add.get("C",0),
    "D_add": counts_add.get("D",0),
    "E_add": counts_add.get("E",0),
    **corr_geo,
    **corr_add
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(TABLE_DIR, "step2_baseline_comparison_summary.csv"), index=False)

print(summary_df.to_string(index=False))

# ----------------------
# Scatter plot
# ----------------------
plt.figure()
plt.scatter(score_geo, score_add, s=10)
plt.xlabel("Geometric Score")
plt.ylabel("Additive Score")
plt.title("Geometric vs Additive")
plt.savefig(os.path.join(FIG_DIR,"scatter_geo_vs_add.png"), dpi=300)
plt.close()