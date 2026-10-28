from pathlib import Path
import pandas as pd
import numpy as np

# ---- CONFIG ----
RESIDUE_CSV = Path("/content/drive/MyDrive/WERM/outputs/runs/residue_r18_mae_yearsplit_20260211_104508/val_preds_best.csv")
COLOR_CSV   = Path("/content/drive/MyDrive/WERM/outputs/color/color_residue_usgs.csv")
TILLAGE_CSV = Path("/content/drive/MyDrive/WERM/outputs/tillage/tillage_residue_usgs.csv")

OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/fused_residue_val.csv")

# Weights (baseline)
W_MOIST = 0.30
W_RES   = 0.20
W_COL   = 0.10
W_TIL   = 0.20
W_TMP   = 0.20

# For now temperature is a constant placeholder risk (set later from GDD)
R_TEMP_CONST = 0.5
# --------------

def risk_class(score):
    # A highest risk, E lowest
    if score >= 0.80: return "A"
    if score >= 0.65: return "B"
    if score >= 0.50: return "C"
    if score >= 0.35: return "D"
    return "E"

def main():
    # Residue preds
    res = pd.read_csv(RESIDUE_CSV)
    # res has photo_id / gt_fR / pred_fR (from our dump script)
    if "photo_id" not in res.columns and "name" in res.columns:
        res = res.rename(columns={"name":"photo_id"})

    res["Residue_pct_pred"] = 100.0 * res["pred_fR"].clip(0,1)
    res["R_residue"] = 1.0 - res["pred_fR"].clip(0,1)

    # Color
    col = pd.read_csv(COLOR_CSV)
    col = col.rename(columns={"name":"photo_id"})

    # Tillage
    til = pd.read_csv(TILLAGE_CSV)
    til = til.rename(columns={"name":"photo_id"})

    # Merge
    df = res.merge(col[["photo_id","R_color"]], on="photo_id", how="inner")
    df = df.merge(til[["photo_id","R_tillage"]], on="photo_id", how="inner")

    # Moisture placeholder (until we fuse plot-level later)
    df["R_moist"] = 0.5
    df["R_temp"]  = R_TEMP_CONST

    df["RiskScore"] = (
        W_MOIST*df["R_moist"] +
        W_RES*df["R_residue"] +
        W_COL*df["R_color"] +
        W_TIL*df["R_tillage"] +
        W_TMP*df["R_temp"]
    ).clip(0,1)

    df["RiskClass"] = df["RiskScore"].apply(risk_class)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, "rows:", len(df))
    print(df["RiskClass"].value_counts())

if __name__ == "__main__":
    main()