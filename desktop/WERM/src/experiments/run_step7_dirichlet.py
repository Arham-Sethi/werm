import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/07_dirichlet_weights"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------
# Columns
# ----------------------
ID_COL = "image"
M_COL = "moisture"
R_COL = "residue"
G_COL = "gdd_norm"
T_COL = "tillage_eff"

# ----------------------
# Settings
# ----------------------
eps = 1e-6
K = 500
dirichlet_alpha = np.array([1.0, 1.0, 1.0, 1.0])  # uniform simplex
np.random.seed(42)

# Base weights (your default)
base_w = {"M":0.30, "G":0.20, "R":0.20, "T":0.30}

# ----------------------
# Helpers
# ----------------------
def spearman_corr(a, b):
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0,1])

def top20_jaccard(a, b):
    k = int(0.2 * len(a))
    A = set(np.argsort(a)[-k:])
    B = set(np.argsort(b)[-k:])
    return len(A & B) / len(A | B)

def robust_scale(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = 1e-12
    z = (x - med) / mad
    return 50 * (np.tanh(z) + 1)

def compute_score(M, U, T, G, wM, wG, wR, wT):
    Mc = np.maximum(M, eps)
    Uc = np.maximum(U, eps)
    Tc = np.maximum(T, eps)
    Gc = np.maximum(G, eps)
    log_r = (
        wM*np.log(Mc) +
        wG*np.log(Gc) +
        wR*np.log(Uc) +
        wT*np.log(Tc)
    )
    return robust_scale(log_r)

def ranks_from_scores(scores):
    # rank 1 = lowest, rank N = highest
    return pd.Series(scores).rank(method="average").to_numpy()

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(INPUT_CSV)
ids = df[ID_COL].values

M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = 1 - R
G = df[G_COL].astype(float).to_numpy()
T = df[T_COL].astype(float).to_numpy()

N = len(ids)

# Base run
score_base = compute_score(M, U, T, G, base_w["M"], base_w["G"], base_w["R"], base_w["T"])
rank_base = ranks_from_scores(score_base)

# Storage
rank_matrix = np.zeros((K, N), dtype=np.float32)
spears = []
jaccs = []

# Sample weights and run
W = np.random.dirichlet(dirichlet_alpha, size=K)  # columns correspond to [M,G,R,T]

for i in range(K):
    wM, wG, wR, wT = W[i]
    score = compute_score(M, U, T, G, wM, wG, wR, wT)
    r = ranks_from_scores(score)
    rank_matrix[i] = r

    spears.append(spearman_corr(score_base, score))
    jaccs.append(top20_jaccard(score_base, score))

spears = np.array(spears)
jaccs = np.array(jaccs)

# Per-image rank stats
rank_mean = rank_matrix.mean(axis=0)
rank_std = rank_matrix.std(axis=0)

topk = int(0.2 * N)
# probability an image is in top20% (rank among top)
top20_prob = (rank_matrix >= (N - topk + 1)).mean(axis=0)

img_stats = pd.DataFrame({
    ID_COL: ids,
    "rank_mean": rank_mean,
    "rank_std": rank_std,
    "top20_prob": top20_prob
}).sort_values("rank_mean", ascending=False)

img_stats_path = os.path.join(TABLE_DIR, "step7_image_rank_stats.csv")
img_stats.to_csv(img_stats_path, index=False)

# Summary metrics
summary = pd.DataFrame([{
    "K": K,
    "dirichlet_alpha": "1,1,1,1",
    "spearman_mean": float(spears.mean()),
    "spearman_p05": float(np.percentile(spears, 5)),
    "spearman_min": float(spears.min()),
    "jaccard_mean": float(jaccs.mean()),
    "jaccard_p05": float(np.percentile(jaccs, 5)),
    "jaccard_min": float(jaccs.min()),
    "robust_top20_count_prob_ge_0.9": int((top20_prob >= 0.9).sum()),
    "uncertain_count_rankstd_ge_50": int((rank_std >= 50).sum())
}])

summary_path = os.path.join(TABLE_DIR, "step7_dirichlet_summary.csv")
summary.to_csv(summary_path, index=False)

print("Saved:", summary_path)
print("Saved:", img_stats_path)
print("\nSUMMARY:\n", summary.to_string(index=False))

# Figure: histogram of rank_std
plt.figure()
plt.hist(rank_std, bins=30)
plt.xlabel("Rank Std Dev across Dirichlet weights")
plt.ylabel("Count of images")
plt.title("Step 07: Rank Uncertainty under Random Weight Sampling")
fig_path = os.path.join(FIG_DIR, "rank_std_hist.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", fig_path)