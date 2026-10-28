import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# ================= CONFIG =================
FUSION_IMG_DIR = Path("/content/drive/MyDrive/WERM/data/fusion")

MOISTURE_CKPT = Path("/content/drive/MyDrive/WERM/outputs/runs/AA_70_30_ep30_20260127_044224/multitask_moisture_reg_01.pt")
RESIDUE_CKPT  = Path("/content/drive/MyDrive/WERM/outputs/runs/residue_r18_strat_logcosh_20260214_142945/best.pt")

GDD_CSV = Path("/content/drive/MyDrive/WERM/outputs/temp/gdd.csv")

MAX_DIR = Path("/content/drive/MyDrive/WERM/data/maxtill")
MIN_DIR = Path("/content/drive/MyDrive/WERM/data/mintill")

OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_results.csv")

IMG_SIZE = 224
T_BASELINE_GDD = 120.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

from tillage_module import TillageConfig, TillageNormalizer, compute_tillage_from_rgb

# ---------- Moisture Model ----------
class MoistureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.reg_head(feats)


# ---------- Residue Model ----------
def build_residue_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model


# ---------- Load GDD ----------
def load_gdd():
    df = pd.read_csv(GDD_CSV)
    total = df["Daily_GDD_Cd"].sum()
    return min(total / T_BASELINE_GDD, 1.0)


# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


# ---------- Main ----------
def main():

    print("Device:", DEVICE)

    # Load models
    moisture_model = MoistureModel().to(DEVICE)
    residue_model = build_residue_model().to(DEVICE)

    moisture_ckpt = torch.load(MOISTURE_CKPT, map_location=DEVICE)
    residue_ckpt = torch.load(RESIDUE_CKPT, map_location=DEVICE)

    moisture_model.load_state_dict(moisture_ckpt["model"], strict=False)
    residue_model.load_state_dict(residue_ckpt["model"], strict=False)

    moisture_model.eval()
    residue_model.eval()

    # ---- Fit Tillage Normalizer ----
    cfg = TillageConfig()
    norm = TillageNormalizer()

    fusion_paths = [p for p in FUSION_IMG_DIR.iterdir()
                if p.suffix.lower() in [".jpg",".png",".jpeg"]]

    all_paths = (
        list(MAX_DIR.glob("*")) +
        list(MIN_DIR.glob("*")) +
        fusion_paths
    )
    norm.fit_from_image_paths(all_paths, cfg)

    print("Tillage normalizer fitted.")
    print("LO:", norm.lo)
    print("HI:", norm.hi)

    G = load_gdd()
    print("Normalized GDD:", G)

    rows = []

    imgs = sorted([p for p in FUSION_IMG_DIR.iterdir()
                   if p.suffix.lower() in [".jpg",".png",".jpeg"]])

    for p in tqdm(imgs, desc="Processing fusion"):

        image = Image.open(p).convert("RGB")

        # Moisture
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            M = float(moisture_model(x).item())

        # Residue
        with torch.no_grad():
            R = float(residue_model(x).item())

        # Tillage (new module)
        rgb_np = np.array(image)
        till_out = compute_tillage_from_rgb(
            rgb_np,
            normalizer=norm,
            cfg=cfg,
            residue_pred=R
        )

        T_raw = till_out["tillage_raw"]
        T_eff = till_out["tillage_eff"]
        T_conf = till_out["tillage_conf"]

        # Biological Fusion
        risk = M * G * (1 - R) * T_eff

        rows.append({
            "image": p.name,
            "moisture": M,
            "residue": R,
            "tillage_raw": T_raw,
            "tillage_eff": T_eff,
            "tillage_conf": T_conf,
            "gdd_norm": G,
            "risk_score": risk
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("risk_score", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved fusion results:", OUT_CSV)
    print("\nTop 10 high risk:")
    print(df.head(10))

    print("\nBottom 10 low risk:")
    print(df.tail(10))


if __name__ == "__main__":
    main()