from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random

IMG_DIR = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/color/color_robustness_c3.csv")

N_SAMPLES = 200          # enough for stats but quick
BRIGHT_FACTORS = [0.8, 1.2]  # -20%, +20%

def list_images(img_dir: Path):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    return sorted([p for p in img_dir.iterdir()
                   if p.suffix in exts and p.is_file()])

def rgb_to_hsv_np(rgb):
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    v = mx
    s = np.where(mx == 0, 0, diff / (mx + 1e-12))
    return s, v

def gray_world(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    means = arr.reshape(-1,3).mean(axis=0)
    target = means.mean()
    scale = target / (means + 1e-12)
    arr = np.clip(arr * scale[None,None,:], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def R_color(img):
    # Run C2 definition: gray-world → HSV darkness
    img_gw = gray_world(img)
    arr = np.asarray(img_gw).astype(np.float32) / 255.0
    _, v = rgb_to_hsv_np(arr)
    D = 1.0 - float(v.mean())
    return D  # unnormalized D_gw

def main():
    imgs = list_images(IMG_DIR)
    random.seed(0)
    sample = random.sample(imgs, min(N_SAMPLES, len(imgs)))

    rows = []
    for p in tqdm(sample, desc="Color robustness C3"):
        img = Image.open(p).convert("RGB")

        base = R_color(img)

        for f in BRIGHT_FACTORS:
            img_b = ImageEnhance.Brightness(img).enhance(f)
            pert = R_color(img_b)
            rows.append({
                "photo_id": p.name,
                "brightness_factor": f,
                "D_gw_base": base,
                "D_gw_pert": pert,
                "abs_delta": abs(pert - base)
            })

    df = pd.DataFrame(rows)

    # Summaries (paper stats)
    mean_delta = float(df["abs_delta"].mean())
    p95_delta = float(df["abs_delta"].quantile(0.95))
    max_delta = float(df["abs_delta"].max())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("N images:", len(sample), "Perturbations per image:", len(BRIGHT_FACTORS))
    print("Mean |Δ|:", mean_delta)
    print("95th percentile |Δ|:", p95_delta)
    print("Max |Δ|:", max_delta)

if __name__ == "__main__":
    main()
