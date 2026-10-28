from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import random
from tqdm import tqdm

# ================= CONFIG =================
ROOT = Path("/content/drive/MyDrive/WERM/data")
MAX_DIR = ROOT / "maxtill"
MIN_DIR = ROOT / "mintill"

# 1) Detailed per-image features (for tuning)
OUT_FEATURES_DRIVE = Path("/content/drive/MyDrive/WERM/outputs/tillage/tillage_features.csv")

# 2) Your requested side-by-side report
OUT_REPORT_DRIVE = Path("/content/drive/MyDrive/WERM/outputs/tillage/tillage_validation_results.csv")

RESIZE_TO = 256
GLCM_LEVELS = 32
RANDOM_SEED = 42

# weights (will tune later)
W_LAP  = 0.0
W_EDGE = 0.0
W_GLCM = 1.0
# ==========================================


def list_images(img_dir: Path):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    return sorted([p for p in img_dir.iterdir() if p.suffix in exts and p.is_file()])


def preprocess_gray(path: Path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    arr = cv2.resize(arr, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float((edges > 0).mean())


def glcm_contrast(gray, levels=32):
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    a = g[:, :-1]
    b = g[:, 1:]

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.ravel(), b.ravel()), 1.0)

    s = M.sum()
    if s > 0:
        M /= s

    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    return float(((i - j) ** 2 * M).sum())


def robust_minmax(x):
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    x = (x - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0, 1), float(lo), float(hi)


def main():
    random.seed(RANDOM_SEED)

    max_imgs = list_images(MAX_DIR)
    min_imgs = list_images(MIN_DIR)

    if len(max_imgs) == 0 or len(min_imgs) == 0:
        raise RuntimeError("Missing images in maxtill or mintill folder.")

    print(f"MaxTill images: {len(max_imgs)}")
    print(f"MinTill images: {len(min_imgs)}")

    rows = []
    for label, img_list in [("maxtill", max_imgs), ("mintill", min_imgs)]:
        for p in tqdm(img_list, desc=f"Processing {label}"):
            gray = preprocess_gray(p)
            lv = laplacian_var(gray)
            ed = edge_density(gray)
            gc = glcm_contrast(gray, levels=GLCM_LEVELS)

            rows.append({
                "path": str(p),
                "name": p.name,
                "label": label,
                "lap_var": lv,
                "edge_den": ed,
                "glcm_contrast": gc
            })

    df = pd.DataFrame(rows)

    # log stabilize laplacian
    df["lap_var_log"] = np.log1p(df["lap_var"].values)

    # pooled normalization
    df["lap_n"], lo1, hi1 = robust_minmax(df["lap_var_log"])
    df["edge_n"], lo2, hi2 = robust_minmax(df["edge_den"])
    df["glcm_n"], lo3, hi3 = robust_minmax(df["glcm_contrast"])

    # Use only GLCM
    raw_score = df["glcm_n"].values

    # Rescale based on class means
    max_mask = df["label"] == "maxtill"
    min_mask = df["label"] == "mintill"

    mean_max = raw_score[max_mask].mean()
    mean_min = raw_score[min_mask].mean()

    scaled = (raw_score - mean_min) / (mean_max - mean_min + 1e-12)
    scaled = np.clip(scaled, 0, 1)

    df["R_tillage"] = scaled


    # Save full features CSV (for weight search)
    OUT_FEATURES_DRIVE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FEATURES_DRIVE, index=False)
    print("\nSaved features CSV:", OUT_FEATURES_DRIVE)

    # Build your report CSV (2 columns + summary)
    max_vals = df[df["label"]=="maxtill"]["R_tillage"].values
    min_vals = df[df["label"]=="mintill"]["R_tillage"].values

    mean_max = float(np.mean(max_vals))
    mean_min = float(np.mean(min_vals))

    n_pairs = min(len(max_vals), len(min_vals))
    max_sample = random.sample(list(max_vals), n_pairs)
    min_sample = random.sample(list(min_vals), n_pairs)
    pair_diffs = [mx - mn for mx, mn in zip(max_sample, min_sample)]
    mean_diff = float(np.mean(pair_diffs))

    print("\n==== SUMMARY ====")
    print("Mean MaxTill:", mean_max)
    print("Mean MinTill:", mean_min)
    print("Mean Pairwise Difference:", mean_diff)

    L = max(len(max_vals), len(min_vals))
    max_col = list(max_vals) + [None]*(L - len(max_vals))
    min_col = list(min_vals) + [None]*(L - len(min_vals))

    out_df = pd.DataFrame({
        "MaxTill_Predictions": max_col,
        "MinTill_Predictions": min_col
    })

    summary = pd.DataFrame({
        "MaxTill_Predictions": ["MEAN_MAXTILL", "MEAN_MINTILL", "MEAN_PAIR_DIFF"],
        "MinTill_Predictions": [mean_max, mean_min, mean_diff]
    })

    final_df = pd.concat([out_df, summary], ignore_index=True)

    OUT_REPORT_DRIVE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_REPORT_DRIVE, index=False)

    print("\nSaved report CSV:", OUT_REPORT_DRIVE)

    print("\nNormalization ranges (pooled):")
    print(f"lap_var_log: {lo1:.4f} – {hi1:.4f}")
    print(f"edge_den   : {lo2:.6f} – {hi2:.6f}")
    print(f"glcm       : {lo3:.4f} – {hi3:.4f}")
    print(f"Weights used: lap={W_LAP}, edge={W_EDGE}, glcm={W_GLCM}")


if __name__ == "__main__":
    main()
