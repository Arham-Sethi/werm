import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/06_noise_robustness"
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
# Base weights
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

def top20_jaccard(a, b):
    k = int(0.2 * len(a))
    A = set(np.argsort(a)[-k:])
    B = set(np.argsort(b)[-k:])
    return len(A & B) / len(A | B)

def compute_score(M, U, T, G):
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

# ----------------------
# Load
# ----------------------
np.random.seed(42)
df = pd.read_csv(INPUT_CSV)

M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = 1 - R
T = df[T_COL].astype(float).to_numpy()
G = df[G_COL].astype(float).to_numpy()

score_base = compute_score(M, U, T, G)

# ----------------------
# Noise levels
# ----------------------
noise_levels = [0.01, 0.03, 0.05]

rows = []

for sigma in noise_levels:
    noise_M = M * (1 + np.random.normal(0, sigma, len(M)))
    noise_R = R * (1 + np.random.normal(0, sigma, len(R)))
    noise_T = T * (1 + np.random.normal(0, sigma, len(T)))
    
    # clip
    noise_M = np.clip(noise_M, 0, 1)
    noise_R = np.clip(noise_R, 0, 1)
    noise_T = np.clip(noise_T, 0, 1)
    
    noise_U = 1 - noise_R
    
    score_noisy = compute_score(noise_M, noise_U, noise_T, G)
    
    rows.append({
        "noise_sigma": sigma,
        "spearman_vs_base": spearman_corr(score_base, score_noisy),
        "top20_jaccard_vs_base": top20_jaccard(score_base, score_noisy)
    })

summary = pd.DataFrame(rows)
summary_path = os.path.join(TABLE_DIR, "step6_noise_summary.csv")
summary.to_csv(summary_path, index=False)

print("Saved:", summary_path)
print(summary.to_string(index=False))