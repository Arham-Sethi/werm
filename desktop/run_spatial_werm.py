"""
WERM Spatial Pipeline — Run in Google Colab with Drive mounted.

Produces: spatial_werm_outputs.csv with columns:
  image_id, moisture_norm, residue_norm, texture_norm, t_eff, gdd_norm, risk_score

Usage (Colab cell):
    from google.colab import drive
    drive.mount('/content/drive')
    %run /content/drive/MyDrive/WERM/run_spatial_werm.py
"""

import math, numpy as np, pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
from tqdm import tqdm

# ======================== PATHS ========================
# Cropped images — upload cropped_spatial/ to Drive, or adjust path
IMG_DIR = Path("/content/drive/MyDrive/WERM/data/cropped_spatial")

MOISTURE_CKPT = Path(
    "/content/drive/MyDrive/WERM/outputs/runs/"
    "AA_70_30_ep30_20260127_044224/multitask_moisture_reg_01.pt"
)
RESIDUE_CKPT = Path(
    "/content/drive/MyDrive/WERM/outputs/runs/"
    "residue_r18_strat_logcosh_20260214_142945/best.pt"
)

OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/spatial_werm_outputs.csv")

# ======================== CONSTANTS ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# GDD: pre-computed from NASA POWER API
# Coords 22.7827, 75.9470 | date 2026-03-18 | Tb=10 | 7-day window
# Total GDD = 129.38, normalized by 120, clamped to 1.0
GDD_NORM = 1.0

# Tillage GLCM
TILL_LEVELS = 32
TILL_RESIZE = 256
TILL_FIT_N = 80
TILL_P_LO = 5
TILL_P_HI = 95
TILL_FLOOR = 0.02
TILL_GAMMA = 0.7
TILL_RESIDUE_DAMP_ALPHA = 1.0

# Fusion weights
WM, WG, WR, WT = 0.30, 0.20, 0.20, 0.30
EPS = 1e-6


# ======================== MODELS ========================
class MoistureRegOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_reg = nn.Linear(in_f, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc_reg(self.backbone(x)))


def load_moisture_model(ckpt_path, device):
    model = MoistureRegOnly().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    remapped = {k: v for k, v in sd.items()
                if k.startswith("backbone.") or k.startswith("fc_reg.")}
    miss, unex = model.load_state_dict(remapped, strict=False)
    print(f"Moisture: loaded ({len(miss)} missing, {len(unex)} unexpected)")
    assert "fc_reg.weight" not in miss
    model.eval()
    return model


def build_residue_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1), nn.Sigmoid())
    return m


def load_residue_model(ckpt_path, device):
    model = build_residue_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"Residue: loaded ({len(miss)} missing, {len(unex)} unexpected)")
    model.eval()
    return model


# ======================== GLCM TEXTURE ========================
def glcm_contrast(gray, levels=32):
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    a, b = g[:, :-1], g[:, 1:]
    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.ravel(), b.ravel()), 1.0)
    M /= M.sum() + 1e-12
    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    return float(((i - j) ** 2 * M).sum())


def tillage_raw(image: Image.Image):
    arr = np.array(image.resize((TILL_RESIZE, TILL_RESIZE)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return glcm_contrast(gray, levels=TILL_LEVELS)


def fit_tillage_norm(img_paths):
    n = min(TILL_FIT_N, len(img_paths))
    if n == 0:
        return 0.6, 18.0
    idxs = np.linspace(0, len(img_paths) - 1, num=n).astype(int)
    raws = []
    for i in idxs:
        try:
            im = Image.open(img_paths[i]).convert("RGB")
            raws.append(tillage_raw(im))
        except Exception:
            continue
    if len(raws) < 5:
        return 0.6, 18.0
    lo = float(np.percentile(raws, TILL_P_LO))
    hi = float(np.percentile(raws, TILL_P_HI))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def norm01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ======================== TRANSFORMS ========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ======================== MAIN ========================
def main():
    print(f"Device: {DEVICE}")

    imgs = sorted([p for p in IMG_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Images: {len(imgs)}")
    assert len(imgs) > 0, f"No images found in {IMG_DIR}"

    # Fit tillage normalizer on this batch
    till_lo, till_hi = fit_tillage_norm(imgs)
    print(f"Tillage normalizer: lo={till_lo:.4f}  hi={till_hi:.4f}")

    # Load models
    moisture_model = load_moisture_model(MOISTURE_CKPT, DEVICE)
    residue_model = load_residue_model(RESIDUE_CKPT, DEVICE)

    print(f"GDD norm (pre-computed): {GDD_NORM}")

    rows = []
    for p in tqdm(imgs, desc="WERM inference"):
        image = Image.open(p).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            M = float(moisture_model(x).item())
            R = float(residue_model(x).item())

        # Texture
        T_raw_val = tillage_raw(image)
        T_norm = norm01(T_raw_val, till_lo, till_hi)

        # Floor + gamma (applied BEFORE damping, per user spec)
        T_norm = TILL_FLOOR + (1.0 - TILL_FLOOR) * (T_norm ** TILL_GAMMA)

        # Residue damping
        T_eff = T_norm * ((1.0 - R) ** TILL_RESIDUE_DAMP_ALPHA)

        # Geometric fusion (log-space)
        M_t = max(M, EPS)
        G_t = max(GDD_NORM, EPS)
        R_t = max(1.0 - R, EPS)
        T_t = max(T_eff, EPS)

        risk_raw = float(np.exp(
            WM * np.log(M_t) +
            WG * np.log(G_t) +
            WR * np.log(R_t) +
            WT * np.log(T_t)
        ))

        rows.append({
            "image_id": p.stem,
            "moisture_norm": round(M, 6),
            "residue_norm": round(R, 6),
            "texture_norm": round(T_norm, 6),
            "t_eff": round(T_eff, 6),
            "gdd_norm": GDD_NORM,
            "risk_raw": risk_raw,
        })

    df = pd.DataFrame(rows)

    # MAD-tanh scaling -> 0-100
    log_r = np.log(df["risk_raw"].values + 1e-9)
    mu = float(np.median(log_r))
    mad = float(np.median(np.abs(log_r - mu)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(log_r) + 1e-6)
    z = (log_r - mu) / sigma
    df["risk_score"] = np.round(50.0 + 50.0 * np.tanh(z / 2.0), 4)

    # Drop intermediate column
    df.drop(columns=["risk_raw"], inplace=True)

    # Sort by risk descending
    df.sort_values("risk_score", ascending=False, inplace=True)

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
    print(f"Score range: {df['risk_score'].min():.2f} – {df['risk_score'].max():.2f}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
