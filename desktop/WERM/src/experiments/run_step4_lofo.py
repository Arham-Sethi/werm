import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/04_lofo_ablation"
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
base_w = {"M":0.30, "G":0.20, "R":0.20, "T":0.30}
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

def compute_geo(M, U, T, G, w):
    Mc = np.maximum(M, eps)
    Uc = np.maximum(U, eps)
    Tc = np.maximum(T, eps)
    Gc = np.maximum(G, eps)
    log_r = (
        w["M"]*np.log(Mc) +
        w["G"]*np.log(Gc) +
        w["R"]*np.log(Uc) +
        w["T"]*np.log(Tc)
    )
    return robust_scale(log_r)

def renormalize(w):
    s = sum(w.values())
    return {k:v/s for k,v in w.items()}

def top20_jaccard(a, b):
    k = int(0.2 * len(a))
    A = set(np.argsort(a)[-k:])
    B = set(np.argsort(b)[-k:])
    return len(A & B) / len(A | B)

# ----------------------
# Load
# ----------------------
df = pd.read_csv(INPUT_CSV)

M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = (1 - R)
T = df[T_COL].astype(float).to_numpy()
G = df[G_COL].astype(float).to_numpy()

# Full model
score_full = compute_geo(M, U, T, G, base_w)
class_full = risk_class(score_full)
full_counts = class_full.value_counts()

rows = []

factors = ["M","R","T","G"]

for f in factors:
    w = base_w.copy()
    w[f] = 0.0
    w = renormalize(w)
    score = compute_geo(M, U, T, G, w)
    cls = risk_class(score)
    cnt = cls.value_counts()
    rows.append({
        "removed_factor": f,
        "spearman_vs_full": spearman_corr(score_full, score),
        "top20_jaccard_vs_full": top20_jaccard(score_full, score),
        "A": int(cnt.get("A",0)),
        "B": int(cnt.get("B",0)),
        "C": int(cnt.get("C",0)),
        "D": int(cnt.get("D",0)),
        "E": int(cnt.get("E",0)),
    })

summary = pd.DataFrame(rows)
summary_path = os.path.join(TABLE_DIR, "step4_lofo_summary.csv")
summary.to_csv(summary_path, index=False)

print("Saved:", summary_path)
print(summary.to_string(index=False))