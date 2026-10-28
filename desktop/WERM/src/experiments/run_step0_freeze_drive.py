import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# INPUT (Drive path)
# -----------------------------
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"

# Column names in your fusion_results_FREEZE.csv
ID_COL = "image"
M_COL  = "moisture"
R_COL  = "residue"
T_COL  = "tillage_eff"
G_COL  = "gdd_norm"

# Weights
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30

# -----------------------------
# OUTPUT DIRS (Drive path)
# -----------------------------
BASE_DIR  = "/content/drive/MyDrive/WERM/experiments/00_freeze"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR   = os.path.join(BASE_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via rank correlation (no scipy needed)."""
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = (np.sqrt((ar**2).sum()) * np.sqrt((br**2).sum()))
    return float((ar * br).sum() / denom) if denom != 0 else float("nan")

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def risk_scale_logmad_tanh(log_risk: np.ndarray) -> np.ndarray:
    """log(risk_raw) -> robust median/MAD -> tanh -> [0,100]"""
    med = np.median(log_risk)
    m = mad(log_risk)
    if m == 0:
        m = 1e-12
    z = (log_risk - med) / m
    y = np.tanh(z)
    return 50.0 * (y + 1.0)

def risk_class(score: np.ndarray) -> pd.Categorical:
    bins = [-1e9, 20, 40, 60, 80, 1e9]
    labels = ["A", "B", "C", "D", "E"]
    return pd.cut(score, bins=bins, labels=labels, right=False)

def run_eps(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    U = 1.0 - df[R_COL].astype(float).to_numpy()

    M = df[M_COL].astype(float).to_numpy()
    T = df[T_COL].astype(float).to_numpy()
    G = df[G_COL].astype(float).to_numpy()

    Mc = np.maximum(M, eps)
    Uc = np.maximum(U, eps)
    Tc = np.maximum(T, eps)
    Gc = np.maximum(G, eps)

    M_clamped = (M < eps)
    U_clamped = (U < eps)
    T_clamped = (T < eps)
    G_clamped = (G < eps)

    log_risk = (wM*np.log(Mc) + wG*np.log(Gc) + wR*np.log(Uc) + wT*np.log(Tc))
    risk_raw = np.exp(log_risk)

    score = risk_scale_logmad_tanh(log_risk)
    cls = risk_class(score)

    out = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "eps": eps,
        "M": M,
        "R": df[R_COL].astype(float).to_numpy(),
        "U": U,
        "T_eff": T,
        "G": G,
        "risk_raw": risk_raw,
        "log_risk": log_risk,
        "risk_score_0_100": score,
        "risk_class": cls.astype(str),
        "M_was_clamped": M_clamped,
        "U_was_clamped": U_clamped,
        "T_was_clamped": T_clamped,
        "G_was_clamped": G_clamped,
    })
    return out

def topk_jaccard(ids_a: np.ndarray, scores_a: np.ndarray,
                 ids_b: np.ndarray, scores_b: np.ndarray,
                 frac: float = 0.20) -> float:
    k_a = int(np.ceil(frac * len(scores_a)))
    k_b = int(np.ceil(frac * len(scores_b)))
    top_a = set(ids_a[np.argsort(scores_a)[-k_a:]])
    top_b = set(ids_b[np.argsort(scores_b)[-k_b:]])
    inter = len(top_a & top_b)
    uni = len(top_a | top_b)
    return inter / uni if uni else float("nan")

def clamp_counts(o: pd.DataFrame):
    return {
        "M_clamped": int(o["M_was_clamped"].sum()),
        "U_clamped": int(o["U_was_clamped"].sum()),
        "T_clamped": int(o["T_was_clamped"].sum()),
        "G_clamped": int(o["G_was_clamped"].sum()),
    }

def class_counts(o: pd.DataFrame):
    vc = o["risk_class"].value_counts()
    return {c: int(vc.get(c, 0)) for c in ["A","B","C","D","E"]}

# -----------------------------
# Main
# -----------------------------
df = pd.read_csv(INPUT_CSV)

needed = [ID_COL, M_COL, R_COL, T_COL, G_COL]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns in CSV: {missing}\n"
        f"Columns found: {list(df.columns)}"
    )

eps1, eps2 = 1e-6, 1e-4
out1 = run_eps(df, eps1)
out2 = run_eps(df, eps2)

# Save outputs to Drive tables folder
out1_path = os.path.join(TABLE_DIR, "risk_eps_1e-6.csv")
out2_path = os.path.join(TABLE_DIR, "risk_eps_1e-4.csv")
out1.to_csv(out1_path, index=False)
out2.to_csv(out2_path, index=False)

# Metrics
spear = spearman_corr(out1["risk_score_0_100"].to_numpy(),
                      out2["risk_score_0_100"].to_numpy())

jac = topk_jaccard(out1[ID_COL].to_numpy(), out1["risk_score_0_100"].to_numpy(),
                   out2[ID_COL].to_numpy(), out2["risk_score_0_100"].to_numpy(),
                   frac=0.20)

cc1, cc2 = clamp_counts(out1), clamp_counts(out2)
bc1, bc2 = class_counts(out1), class_counts(out2)

summary = pd.DataFrame([
    {"eps": "1e-6", "spearman_vs_1e-6": "", "top20_jaccard_vs_1e-6": "", **cc1, **bc1},
    {"eps": "1e-4", "spearman_vs_1e-6": spear, "top20_jaccard_vs_1e-6": jac, **cc2, **bc2},
])

summary_path = os.path.join(TABLE_DIR, "step0_metrics_summary.csv")
summary.to_csv(summary_path, index=False)

# Scatter plot to Drive figures folder
plt.figure()
plt.scatter(out1["risk_score_0_100"], out2["risk_score_0_100"], s=10)
plt.xlabel("risk_score_0_100 (eps=1e-6)")
plt.ylabel("risk_score_0_100 (eps=1e-4)")
plt.title("Step 0: Epsilon Stability Scatter")
fig_path = os.path.join(FIG_DIR, "step0_eps_rank_scatter.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("DONE ✅")
print("Saved:", out1_path)
print("Saved:", out2_path)
print("Saved:", summary_path)
print("Saved:", fig_path)

# Quick existence check
print("\nExists checks:")
for p in [out1_path, out2_path, summary_path, fig_path]:
    print(p, "->", os.path.exists(p))

print("\nSpearman:", spear)
print("Top-20% Jaccard:", jac)
print("Clamp counts eps=1e-6:", cc1)
print("Clamp counts eps=1e-4:", cc2)
print("A–E eps=1e-6:", bc1)
print("A–E eps=1e-4:", bc2)