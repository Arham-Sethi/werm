import numpy as np
import pandas as pd

CSV = "/content/drive/MyDrive/WERM/outputs/tillage/tillage_features.csv"

df = pd.read_csv(CSV)

needed = ["label", "lap_var_log", "edge_den", "glcm_contrast"]
for c in needed:
    if c not in df.columns:
        raise RuntimeError(f"Missing column {c}. Re-run validate_tillage_texture.py first.")

def robust_minmax(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    x = (x - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0, 1)

lap_n = robust_minmax(df["lap_var_log"].values.astype(float))
edge_n = robust_minmax(df["edge_den"].values.astype(float))
glcm_n = robust_minmax(df["glcm_contrast"].values.astype(float))

y = (df["label"].astype(str).values == "maxtill").astype(int)

best = None
grid = np.arange(0.0, 1.01, 0.05)

for w_lap in grid:
    for w_edge in grid:
        w_glcm = 1.0 - w_lap - w_edge
        if w_glcm < 0:
            continue

        score = np.clip(w_lap*lap_n + w_edge*edge_n + w_glcm*glcm_n, 0, 1)
        max_mean = float(score[y==1].mean())
        min_mean = float(score[y==0].mean())
        diff = max_mean - min_mean

        # objective: separate + push max up + push min down
        obj = diff + 0.25*max_mean - 0.25*min_mean

        if best is None or obj > best["obj"]:
            best = {
                "w_lap": float(w_lap),
                "w_edge": float(w_edge),
                "w_glcm": float(w_glcm),
                "max_mean": max_mean,
                "min_mean": min_mean,
                "diff": diff,
                "obj": float(obj),
            }

print("BEST:", best)
