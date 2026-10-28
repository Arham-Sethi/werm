from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import time
from tqdm import tqdm

# ================= CONFIG =================
IMG_DIR = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
OUT_LOCAL = Path("/content/tillage_residue_usgs.csv")
OUT_DRIVE = Path("/content/drive/MyDrive/WERM/outputs/tillage/tillage_residue_usgs.csv")

RESIZE_TO = 256   # resize for speed (critical)
GLCM_LEVELS = 32
# ==========================================


def list_images(img_dir: Path):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    return sorted([p for p in img_dir.iterdir()
                   if p.suffix in exts and p.is_file()])


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float((edges > 0).mean())


def glcm_contrast(gray, levels=32):
    # Quantize grayscale
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)

    # Horizontal adjacency
    a = g[:, :-1]
    b = g[:, 1:]

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.ravel(), b.ravel()), 1.0)

    M_sum = M.sum()
    if M_sum > 0:
        M /= M_sum

    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    contrast = ((i - j) ** 2 * M).sum()

    return float(contrast)


def robust_minmax(x):
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    x = (x - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0, 1), float(lo), float(hi)


def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    imgs = list_images(IMG_DIR)
    print(f"\nFound {len(imgs)} images.")
    print("Starting texture extraction...\n")

    rows = []
    start_time = time.time()

    for i, p in enumerate(tqdm(imgs, desc="Processing images")):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img)

            # Resize for speed
            arr = cv2.resize(arr, (RESIZE_TO, RESIZE_TO),
                             interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            lv = laplacian_var(gray)
            ed = edge_density(gray)
            gc = glcm_contrast(gray, levels=GLCM_LEVELS)

            rows.append({
                "path": str(p),
                "name": p.name,
                "lap_var": lv,
                "edge_den": ed,
                "glcm_contrast": gc
            })

        except Exception as e:
            print(f"\nSkipping {p.name}: {e}")

        # Heartbeat log every 50 images
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"\nProcessed {i+1}/{len(imgs)} | Elapsed: {elapsed:.1f}s")

    print("\nFeature extraction complete.")
    print("Normalizing features...\n")

    df = pd.DataFrame(rows)

    # Normalize each feature robustly
    df["lap_n"], lo1, hi1 = robust_minmax(df["lap_var"].values)
    df["edge_n"], lo2, hi2 = robust_minmax(df["edge_den"].values)
    df["glcm_n"], lo3, hi3 = robust_minmax(df["glcm_contrast"].values)

    # Weighted looseness index (tunable)
    df["R_tillage"] = np.clip(
        0.45 * df["lap_n"] +
        0.35 * df["edge_n"] +
        0.20 * df["glcm_n"],
        0, 1
    )

    print("Normalization complete.")
    print(f"Laplacian percentile range: {lo1:.4f} – {hi1:.4f}")
    print(f"Edge density percentile range: {lo2:.4f} – {hi2:.4f}")
    print(f"GLCM contrast percentile range: {lo3:.4f} – {hi3:.4f}")

    # Save locally first (Drive-safe)
    df.to_csv(OUT_LOCAL, index=False)
    print(f"\nSaved locally: {OUT_LOCAL}")

    # Copy to Drive
    OUT_DRIVE.parent.mkdir(parents=True, exist_ok=True)
    OUT_DRIVE.write_bytes(OUT_LOCAL.read_bytes())
    print(f"Copied to Drive: {OUT_DRIVE}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
