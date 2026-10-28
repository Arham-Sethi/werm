import os
import numpy as np
import pandas as pd

INPUT_CSV = "/content/drive/MyDrive/WERM/experiments/00_freeze/fusion_results_FREEZE.csv"
OUT_DIR = "/content/drive/MyDrive/WERM/experiments/01_residue_damping/tables"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns (edit ONLY if your CSV uses different names)
ID_COL = "image"
M_COL = "moisture"
R_COL = "residue"
G_COL = "gdd_norm"
T_NORM_COL = "tillage_norm"   # needed for alpha=0
# If you already have alpha=1 T_eff in the CSV, you can use it as a check
T_EFF_COL = "tillage_eff"      # optional; if missing, alpha=1 will be computed from T_norm

# Weights
wM, wG, wR, wT = 0.30, 0.20, 0.20, 0.30
eps = 1e-6

df = pd.read_csv(INPUT_CSV)

# Safety checks
for c in [ID_COL, M_COL, R_COL, G_COL, T_NORM_COL]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)}")

M = df[M_COL].astype(float).to_numpy()
R = df[R_COL].astype(float).to_numpy()
U = 1.0 - R
G = df[G_COL].astype(float).to_numpy()
Tnorm = df[T_NORM_COL].astype(float).to_numpy()

def clamp(x): 
    return np.maximum(x, eps)

def compute_for_alpha(alpha: float):
    Teff = Tnorm * (clamp(U) ** alpha)   # U clamped so power is safe
    Mc = clamp(M)
    Uc = clamp(U)
    Tc = clamp(Teff)
    Gc = clamp(G)

    cM = wM * np.log(Mc)
    cG = wG * np.log(Gc)
    cR = wR * np.log(Uc)
    cT = wT * np.log(Tc)

    log_risk = cM + cG + cR + cT

    # Variances (population variance for consistency)
    var_log_risk = float(np.var(log_risk))
    var_cM = float(np.var(cM))
    var_cG = float(np.var(cG))
    var_cR = float(np.var(cR))
    var_cT = float(np.var(cT))

    # Percent contributions (avoid divide by zero)
    denom = var_log_risk if var_log_risk > 0 else 1e-12
    pct = {
        "pct_cM": var_cM / denom,
        "pct_cG": var_cG / denom,
        "pct_cR": var_cR / denom,
        "pct_cT": var_cT / denom,
    }

    out = {
        "alpha": alpha,
        "Var_log_risk": var_log_risk,
        "Var_cM": var_cM,
        "Var_cG": var_cG,
        "Var_cR": var_cR,
        "Var_cT": var_cT,
        **pct
    }
    return out

rows = [compute_for_alpha(0.0), compute_for_alpha(1.0)]
summary = pd.DataFrame(rows)

# Save
out_path = os.path.join(OUT_DIR, "variance_alpha0_alpha1.csv")
summary.to_csv(out_path, index=False)

print("Saved:", out_path)
print("\nFULL SUMMARY:\n")
print(summary.to_string(index=False))