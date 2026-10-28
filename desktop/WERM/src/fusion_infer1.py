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
OUT_CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_results.csv")

IMG_SIZE = 224
T_BASELINE_GDD = 120.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fit tillage normalization from the fusion set (fast)
TILL_FIT_N = 60         # sample images to fit contrast range
TILL_LEVELS = 32
TILL_RESIZE = 256

# Residue-damping to reduce straw-edge inflating tillage
TILL_RESIDUE_DAMP_ALPHA = 1.0
# ==========================================


import torch
import torch.nn as nn
from torchvision import models

class MoistureRegOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        y = self.fc_reg(feats)
        return torch.sigmoid(y)


def load_moisture_model(ckpt_path, device):
    model = MoistureRegOnly().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Keep backbone + fc_reg only
    remapped = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            remapped[k] = v
        elif k.startswith("fc_reg."):
            remapped[k] = v
        # Drop fc_cls entirely

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    print("Moisture load -> missing:", len(missing), "unexpected:", len(unexpected))

    assert "fc_reg.weight" not in missing and "fc_reg.bias" not in missing, \
        "fc_reg did not load correctly."

    model.eval()
    return model


# ---------- Residue model ----------
def build_residue_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 1),
        nn.Sigmoid()
    )
    return m


def load_residue_model(ckpt_path: Path, device: str):
    model = build_residue_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Residue load -> missing:", len(missing), "unexpected:", len(unexpected))
    model.eval()
    return model


# ---------- Tillage: GLCM contrast ----------
def glcm_contrast(gray, levels=32):
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    a = g[:, :-1]
    b = g[:, 1:]
    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.ravel(), b.ravel()), 1.0)
    M /= (M.sum() + 1e-12)
    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    return float(((i - j) ** 2 * M).sum())


def tillage_raw_from_pil(image):
    arr = np.array(image.resize((TILL_RESIZE, TILL_RESIZE)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return glcm_contrast(gray, levels=TILL_LEVELS)


def fit_tillage_norm(img_paths, n_fit=60):
    # sample evenly to avoid bias
    if len(img_paths) == 0:
        return 0.6, 18.0

    idxs = np.linspace(0, len(img_paths) - 1, num=min(n_fit, len(img_paths))).astype(int)
    raws = []
    for i in idxs:
        p = img_paths[i]
        try:
            im = Image.open(p).convert("RGB")
            raws.append(tillage_raw_from_pil(im))
        except Exception:
            continue

    if len(raws) < 10:
        # fallback
        return 0.6, 18.0

    lo = float(np.percentile(raws, 2))
    hi = float(np.percentile(raws, 98))
    # guard against degenerate
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def tillage_norm_from_raw(raw, lo, hi):
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


# ---------- GDD ----------
def load_gdd():
    df = pd.read_csv(GDD_CSV)
    total = float(df["Daily_GDD_Cd"].sum())
    return min(total / T_BASELINE_GDD, 1.0), total


# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def main():
    print("Device:", DEVICE)

    imgs = sorted([p for p in FUSION_IMG_DIR.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]])
    print("Images:", len(imgs))

    # fit tillage normalization from your actual fusion set
    till_lo, till_hi = fit_tillage_norm(imgs, n_fit=TILL_FIT_N)
    print("Tillage normalizer fitted.")
    print("LO:", till_lo)
    print("HI:", till_hi)

    moisture_model = load_moisture_model(MOISTURE_CKPT, DEVICE)
    residue_model  = load_residue_model(RESIDUE_CKPT, DEVICE)

    G_norm, G_total = load_gdd()
    print("GDD total:", G_total, "| Normalized GDD:", G_norm)

    rows = []
    for p in tqdm(imgs, desc="Processing fusion"):
        image = Image.open(p).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            M = float(moisture_model(x).item())
            R = float(residue_model(x).item())

        T_raw = tillage_raw_from_pil(image)
        T_norm = tillage_norm_from_raw(T_raw, till_lo, till_hi)

        # damp tillage when residue is high (straw edges ≠ tillage)
        T_eff = T_norm * ((1.0 - R) ** TILL_RESIDUE_DAMP_ALPHA)

        risk = M * G_norm * (1.0 - R) * T_eff

        rows.append({
            "image": p.name,
            "moisture": M,
            "residue": R,
            "tillage_raw": T_raw,
            "tillage_norm": T_norm,
            "tillage_eff": T_eff,
            "gdd_norm": G_norm,
            "risk_score": risk
        })

    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved fusion results:", OUT_CSV)

    print("\nRanges:")
    print(" moisture:", df["moisture"].min(), df["moisture"].max())
    print(" residue :", df["residue"].min(), df["residue"].max())
    print(" till_raw:", df["tillage_raw"].min(), df["tillage_raw"].max())
    print(" till_eff:", df["tillage_eff"].min(), df["tillage_eff"].max())
    print(" risk    :", df["risk_score"].min(), df["risk_score"].max())

    print("\nTop 10 high risk:")
    print(df.head(10)[["image","moisture","residue","tillage_norm","tillage_eff","gdd_norm","risk_score"]])

    print("\nBottom 10 low risk:")
    print(df.tail(10)[["image","moisture","residue","tillage_norm","tillage_eff","gdd_norm","risk_score"]])


if __name__ == "__main__":
    main()