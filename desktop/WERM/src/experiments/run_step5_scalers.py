import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/05_scaling_comparison"
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
# Weights + eps
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

def risk_class(score):
    bins = [-1e9,20,40,60,80,1e9]
    labels = ["A","B","C","D","E"]
    return pd.cut(score, bins=bins, labels=labels, right=False)

def entropy(x, bins=10):
    hist, _ = np.histogram(x, bins=bins)
    p = hist / np.sum(hist)
    p = p[p>0]
    return float(-np.sum(p*np.log(p)))

def top20_jaccard(a, b):
    k = int(0.2 * len(a))
    A = set(np.argsort(a)[-k:])
    B = set(np.argsort(b)[-k:])
    return len(A & B) / len(A | B)

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

# ---- Scalers (all output 0..100) ----
def scale_mad_tanh(x):
    med = np.median(x)
    m = mad(x)
    if m == 0: m = 1e-12
    z = (x - med) / m
    return 50*(np.tanh(z)+1)

def scale_z_tanh(x):
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0: sd = 1e-12
    z = (x - mu) / sd
    return 50*(np.tanh(z)+1)

def scale_minmax(x):
    lo, hi = np.min(x), np.max(x)
    if hi - lo == 0: return np.full_like(x, 50.0)
    y = (x - lo) / (hi - lo)
    return 100*y

def scale_pct_clip(x, p_lo=5, p_hi=95):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi - lo == 0: return np.full_like(x, 50.0)
    xc = np.clip(x, lo, hi)
    y = (xc - lo) / (hi - lo)
    return 100*y

# ----------------------
# Load + compute log-risk
# ----------------------
df = pd.read_csv(INPUT_CSV)
M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = 1 - R
G = df[G_COL].astype(float).to_numpy()
T = df[T_COL].astype(float).to_numpy()

Mc = np.maximum(M, eps)
Uc = np.maximum(U, eps)
Gc = np.maximum(G, eps)
Tc = np.maximum(T, eps)

log_risk = (
    wM*np.log(Mc) +
    wG*np.log(Gc) +
    wR*np.log(Uc) +
    wT*np.log(Tc)
)

# ----------------------
# Apply scalers
# ----------------------
scores = {
    "mad_tanh": scale_mad_tanh(log_risk),  # baseline
    "z_tanh":   scale_z_tanh(log_risk),
    "minmax":   scale_minmax(log_risk),
    "pctclip":  scale_pct_clip(log_risk, 5, 95)
}

# ----------------------
# Save row-level scores
# ----------------------
row = pd.DataFrame({ID_COL: df[ID_COL]})
for k,v in scores.items():
    row[f"score_{k}"] = v
    row[f"class_{k}"] = risk_class(v).astype(str)

row_path = os.path.join(TABLE_DIR, "step5_scores_all_scalers.csv")
row.to_csv(row_path, index=False)

# ----------------------
# Metrics vs baseline
# ----------------------
base = scores["mad_tanh"]
rows = []
for name, sc in scores.items():
    cls = risk_class(sc).value_counts()
    rows.append({
        "scaler": name,
        "entropy": entropy(sc),
        "A": int(cls.get("A",0)),
        "B": int(cls.get("B",0)),
        "C": int(cls.get("C",0)),
        "D": int(cls.get("D",0)),
        "E": int(cls.get("E",0)),
        "spearman_vs_mad_tanh": spearman_corr(base, sc) if name != "mad_tanh" else 1.0,
        "top20_jaccard_vs_mad_tanh": top20_jaccard(base, sc) if name != "mad_tanh" else 1.0
    })

summary = pd.DataFrame(rows)
summary_path = os.path.join(TABLE_DIR, "step5_scaler_summary.csv")
summary.to_csv(summary_path, index=False)
print("Saved:", row_path)
print("Saved:", summary_path)
print("\n", summary.to_string(index=False))

# ----------------------
# Figures: 1 histogram per scaler
# ----------------------
for name, sc in scores.items():
    plt.figure()
    plt.hist(sc, bins=20)
    plt.title(f"Risk Score Histogram - {name}")
    plt.xlabel("score (0-100)")
    plt.ylabel("count")
    plt.savefig(os.path.join(FIG_DIR, f"hist_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()