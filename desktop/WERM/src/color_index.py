from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image

# ---- CONFIG ----
# Point this to ANY image folder you want to compute color on.
# Start with residue dataset because it's standardized.
IMG_DIR = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/color/color_residue_usgs.csv")
# --------------

def list_images(img_dir: Path):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    return sorted([p for p in img_dir.iterdir() if p.suffix in exts and p.is_file()])

def rgb_to_hsv_np(rgb):
    # rgb: HxWx3 in [0,1]
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    # Hue (not needed), Saturation, Value
    v = mx
    s = np.where(mx == 0, 0, diff / (mx + 1e-12))
    return s, v

def gray_world(img):
    # simple illumination normalization: scale each channel to same mean
    arr = np.asarray(img).astype(np.float32) / 255.0
    means = arr.reshape(-1,3).mean(axis=0)
    target = means.mean()
    scale = target / (means + 1e-12)
    arr = np.clip(arr * scale[None,None,:], 0, 1)
    return (arr * 255).astype(np.uint8)

def compute_indices(img: Image.Image):
    # raw
    arr = np.asarray(img).astype(np.float32) / 255.0
    s, v = rgb_to_hsv_np(arr)
    D_raw = 1.0 - float(v.mean())  # darkness

    # gray-world normalized
    img_gw = Image.fromarray(gray_world(img))
    arr2 = np.asarray(img_gw).astype(np.float32) / 255.0
    s2, v2 = rgb_to_hsv_np(arr2)
    D_gw = 1.0 - float(v2.mean())

    return D_raw, D_gw

def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)  # robust
    x = (x - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0, 1), float(lo), float(hi)

def main():
    imgs = list_images(IMG_DIR)
    rows = []
    for p in imgs:
        try:
            img = Image.open(p).convert("RGB")
            d_raw, d_gw = compute_indices(img)
            rows.append({"path": str(p), "name": p.name, "D_raw": d_raw, "D_gw": d_gw})
        except Exception as e:
            rows.append({"path": str(p), "name": p.name, "D_raw": None, "D_gw": None, "error": str(e)})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["D_raw","D_gw"]).reset_index(drop=True)

    df["D_raw_norm"], lo1, hi1 = minmax_norm(df["D_raw"].values)
    df["D_gw_norm"],  lo2, hi2 = minmax_norm(df["D_gw"].values)

    # Choose the robust one as your main index
    df["R_color"] = df["D_gw_norm"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Images processed:", len(df))
    print("D_raw percentiles (2,98):", lo1, hi1)
    print("D_gw  percentiles (2,98):", lo2, hi2)

if __name__ == "__main__":
    main()
