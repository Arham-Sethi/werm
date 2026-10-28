import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths ----
INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"

BASE_DIR = "/content/drive/MyDrive/WERM/experiments/01_residue_damping"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Columns ----
ID_COL = "image"
M_COL = "moisture"
R_COL = "residue"
T_NORM_COL = "tillage_norm"   # change if needed
G_COL = "gdd_norm"

# ---- Weights ----
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30
eps = 1e-6

# ---- Helpers ----
def spearman_corr(a, b):
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ar, br)[0,1]

def risk_scale(log_risk):
    med = np.median(log_risk)
    mad = np.median(np.abs(log_risk - med))
    if mad == 0:
        mad = 1e-12
    z = (log_risk - med) / mad
    y = np.tanh(z)
    return 50 * (y + 1)

def risk_class(score):
    bins = [-1e9,20,40,60,80,1e9]
    labels = ["A","B","C","D","E"]
    return pd.cut(score, bins=bins, labels=labels, right=False)

# ---- Load ----
df = pd.read_csv(INPUT_CSV)

alphas = [0,0.5,1,1.5,2]

results = []
risk_reference = None

for alpha in alphas:
    U = 1 - df[R_COL].values
    T_norm = df[T_NORM_COL].values
    
    T_eff = T_norm * (U ** alpha)
    
    Mc = np.maximum(df[M_COL].values, eps)
    Uc = np.maximum(U, eps)
    Tc = np.maximum(T_eff, eps)
    Gc = np.maximum(df[G_COL].values, eps)
    
    log_risk = (
        wM*np.log(Mc) +
        wG*np.log(Gc) +
        wR*np.log(Uc) +
        wT*np.log(Tc)
    )
    
    risk_raw = np.exp(log_risk)
    score = risk_scale(log_risk)
    cls = risk_class(score)
    
    if alpha == 1:
        risk_reference = score
    
    if risk_reference is not None:
        spear = spearman_corr(risk_reference, score)
    else:
        spear = np.nan
    
    top_k = int(0.2 * len(score))
    ref_top = set(np.argsort(risk_reference)[-top_k:]) if risk_reference is not None else set()
    cur_top = set(np.argsort(score)[-top_k:])
    
    jaccard = len(ref_top & cur_top) / len(ref_top | cur_top) if ref_top else np.nan
    
    corr_norm_R = np.corrcoef(T_norm, df[R_COL].values)[0,1]
    corr_eff_R = np.corrcoef(T_eff, df[R_COL].values)[0,1]
    
    counts = cls.value_counts()
    
    results.append({
        "alpha": alpha,
        "spearman_vs_alpha1": spear,
        "top20_jaccard": jaccard,
        "corr_Tnorm_R": corr_norm_R,
        "corr_Teff_R": corr_eff_R,
        "A": counts.get("A",0),
        "B": counts.get("B",0),
        "C": counts.get("C",0),
        "D": counts.get("D",0),
        "E": counts.get("E",0)
    })

summary = pd.DataFrame(results)
summary.to_csv(os.path.join(TABLE_DIR,"alpha_sweep_summary.csv"), index=False)

print(summary)