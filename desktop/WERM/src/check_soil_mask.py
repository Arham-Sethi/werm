import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

SRC = Path("/content/drive/MyDrive/WERM/src")
sys.path.insert(0, str(SRC))

from tillage_module import TillageConfig, build_masks

MAXTILL_DIR = Path("/content/drive/MyDrive/WERM/data/maxtill")
MINTILL_DIR = Path("/content/drive/MyDrive/WERM/data/mintill")
FUSION_DIR  = Path("/content/drive/MyDrive/WERM/data/fusion")

N_FUSION_SAMPLES = 30
OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/tillage/mask_check.csv")


def list_imgs(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix in exts])


def run_on_folder(name: str, paths, cfg: TillageConfig):
    rows = []
    for p in tqdm(paths, desc=f"Mask check: {name}"):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(rgb, (cfg.resize_to, cfg.resize_to), interpolation=cv2.INTER_AREA)

        masks = build_masks(arr, cfg)
        soil = masks["soil_mask"]
        residue_struct = masks["residue_struct_mask"]
        veg = masks["veg_mask"]
        shadow = masks["shadow_mask"]

        tot = soil.size
        rows.append({
            "split": name,
            "image": p.name,
            "soil_frac": float(soil.sum() / tot),
            "residue_struct_frac": float(residue_struct.sum() / tot),
            "veg_frac": float(veg.sum() / tot),
            "shadow_frac": float(shadow.sum() / tot),
        })
    return rows


def summarize(df, split):
    d = df[df["split"] == split]
    if len(d) == 0:
        print(f"[{split}] no rows")
        return
    print(f"\n==== {split} ({len(d)} images) ====")
    for col in ["soil_frac", "residue_struct_frac", "veg_frac", "shadow_frac"]:
        print(f"{col}: mean={d[col].mean():.3f}  p10={d[col].quantile(0.10):.3f}  p50={d[col].quantile(0.50):.3f}  p90={d[col].quantile(0.90):.3f}")


def main():
    cfg = TillageConfig(
        resize_to=256,
        roi_frac=0.75,
        min_soil_frac=0.35,
        use_log_glcm=True,
        tophat_kernel=9,
        tophat_thresh=18,
        edge_thresh=80,
    )

    max_paths = list_imgs(MAXTILL_DIR)
    min_paths = list_imgs(MINTILL_DIR)
    fusion_paths = list_imgs(FUSION_DIR)[:N_FUSION_SAMPLES]

    rows = []
    rows += run_on_folder("maxtill", max_paths, cfg)
    rows += run_on_folder("mintill", min_paths, cfg)
    rows += run_on_folder("fusion_sample", fusion_paths, cfg)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    summarize(df, "mintill")
    summarize(df, "maxtill")
    summarize(df, "fusion_sample")

    if len(df[df["split"] == "mintill"]) > 0:
        m = df[df["split"] == "mintill"]["soil_frac"].mean()
        if m < 0.75:
            print("\n[WARNING] mintill soil_frac mean < 0.75. Residue-structure mask still too aggressive.")
            print("Fix knobs: increase tophat_thresh OR increase edge_thresh OR reduce tophat_kernel.")
        else:
            print("\n[OK] mintill soil_frac looks sane.")


if __name__ == "__main__":
    main()