from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_ROOT = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
CSV_PATH  = DATA_ROOT / "labels.csv"
OUT_CSV   = Path("/content/drive/MyDrive/WERM/outputs/residue_classical/val_preds_best.csv")

TRAIN_YEARS = [2015, 2016, 2017, 2019]
VAL_YEARS   = [2021, 2022]

def year_only(x):
    return int(str(x).split("/")[-1])

def feats_from_image(p: Path):
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img)

    # resize small for speed
    arr = cv2.resize(arr, (256,256), interpolation=cv2.INTER_AREA)

    rgb = arr.astype(np.float32) / 255.0
    mean_rgb = rgb.reshape(-1,3).mean(axis=0)
    std_rgb  = rgb.reshape(-1,3).std(axis=0)

    hsv = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,0] /= 179.0
    hsv[...,1] /= 255.0
    hsv[...,2] /= 255.0
    mean_hsv = hsv.reshape(-1,3).mean(axis=0)
    std_hsv  = hsv.reshape(-1,3).std(axis=0)

    gray = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_den = (edges > 0).mean()

    return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv, [lap_var, edge_den]]).astype(np.float32)

def main():
    df = pd.read_csv(CSV_PATH)
    df["year_only"] = df["year"].apply(year_only)
    df["photo_id"] = df["photo_id"].astype(str).str.strip()

    # keep only existing files
    df["exists"] = df["photo_id"].apply(lambda n: (DATA_ROOT / n).exists())
    df = df[df["exists"]].copy()

    train_df = df[df["year_only"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val_df   = df[df["year_only"].isin(VAL_YEARS)].reset_index(drop=True)

    print("Train:", len(train_df), "Val:", len(val_df))

    Xtr = np.stack([feats_from_image(DATA_ROOT / n) for n in train_df["photo_id"]])
    ytr = train_df["fR"].astype(float).to_numpy()

    Xva = np.stack([feats_from_image(DATA_ROOT / n) for n in val_df["photo_id"]])
    yva = val_df["fR"].astype(float).to_numpy()

    model = Ridge(alpha=10.0, random_state=0)
    model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    pred = np.clip(pred, 0, 1)

    mae = mean_absolute_error(yva, pred)
    rmse = np.sqrt(mean_squared_error(yva, pred))
    r2 = r2_score(yva, pred)

    print("Classical Ridge: MAE=", mae, "RMSE=", rmse, "R2=", r2)

    out = pd.DataFrame({
        "photo_id": val_df["photo_id"].tolist(),
        "gt_fR": yva.tolist(),
        "pred_fR": pred.tolist(),
        "abs_err": np.abs(pred - yva).tolist(),
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
