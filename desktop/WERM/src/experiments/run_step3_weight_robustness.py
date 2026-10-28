import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Paths
# ----------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
BASE_DIR = "/content/drive/MyDrive/WERM/experiments/03_weight_robustness"
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
T_COL = "tillage_eff"   # alpha=1 damping already used

# ----------------------
# Defaults
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

def compute_geo_scores(M, U, T, G, w):
    # clamp for log
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

def top20_jaccard(scores_a, scores_b):
    k = int(0.2 * len(scores_a))
    A = set(np.argsort(scores_a)[-k:])
    B = set(np.argsort(scores_b)[-k:])
    return len(A & B) / len(A | B)

def normalize_weights(w):
    s = sum(w.values())
    return {k: v/s for k,v in w.items()}

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(INPUT_CSV)
M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = (1 - R)
T = df[T_COL].astype(float).to_numpy()
G = df[G_COL].astype(float).to_numpy()

# Base scores
score_base = compute_geo_scores(M, U, T, G, base_w)
class_base = risk_class(score_base)
base_counts = class_base.value_counts()

# ----------------------
# Subtest 1: Equal weights
# ----------------------
eq_w = normalize_weights({"M":0.25,"G":0.25,"R":0.25,"T":0.25})
score_eq = compute_geo_scores(M, U, T, G, eq_w)
class_eq = risk_class(score_eq)
eq_counts = class_eq.value_counts()

eq_metrics = {
    "test": "equal_weights",
    "wM": eq_w["M"], "wG": eq_w["G"], "wR": eq_w["R"], "wT": eq_w["T"],
    "spearman_vs_base": spearman_corr(score_base, score_eq),
    "top20_jaccard_vs_base": top20_jaccard(score_base, score_eq),
    "A": int(eq_counts.get("A",0)), "B": int(eq_counts.get("B",0)),
    "C": int(eq_counts.get("C",0)), "D": int(eq_counts.get("D",0)),
    "E": int(eq_counts.get("E",0)),
}

rows = [eq_metrics]

# ----------------------
# Subtest 2: One-at-a-time sweeps
# ----------------------
deltas = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
keys = ["M","G","R","T"]

for key in keys:
    for d in deltas:
        w = base_w.copy()
        w[key] = w[key] + d
        # if any weight becomes <=0, skip (invalid)
        if any(v <= 0 for v in w.values()):
            continue
        w = normalize_weights(w)
        score = compute_geo_scores(M, U, T, G, w)
        cls = risk_class(score)
        cnt = cls.value_counts()
        rows.append({
            "test": f"sweep_{key}_{d:+.2f}",
            "wM": w["M"], "wG": w["G"], "wR": w["R"], "wT": w["T"],
            "spearman_vs_base": spearman_corr(score_base, score),
            "top20_jaccard_vs_base": top20_jaccard(score_base, score),
            "A": int(cnt.get("A",0)), "B": int(cnt.get("B",0)),
            "C": int(cnt.get("C",0)), "D": int(cnt.get("D",0)),
            "E": int(cnt.get("E",0)),
        })

summary = pd.DataFrame(rows)
summary_path = os.path.join(TABLE_DIR, "step3_weight_robustness_summary.csv")
summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)
print(summary.to_string(index=False))

# ----------------------
# Figure: Spearman vs sweep (simple)
# ----------------------
plt.figure()
mask = summary["test"].str.startswith("sweep_")
plt.plot(np.arange(mask.sum()), summary.loc[mask, "spearman_vs_base"].values, marker="o")
plt.ylabel("Spearman vs base")
plt.xlabel("Sweep runs (in order)")
plt.title("Weight Sweep Robustness (Spearman)")
plt.savefig(os.path.join(FIG_DIR, "spearman_weight_sweeps.png"), dpi=300, bbox_inches="tight")
plt.close()