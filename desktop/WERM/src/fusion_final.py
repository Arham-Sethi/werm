# /content/drive/MyDrive/WERM/src/fusion_final_AE.py

import math
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
from tqdm import tqdm

# ================= CONFIG =================
FUSION_IMG_DIR = Path("/content/drive/MyDrive/WERM/data/fusion")

MOISTURE_CKPT = Path("/content/drive/MyDrive/WERM/outputs/runs/AA_70_30_ep30_20260127_044224/multitask_moisture_reg_01.pt")
RESIDUE_CKPT  = Path("/content/drive/MyDrive/WERM/outputs/runs/residue_r18_strat_logcosh_20260214_142945/best.pt")

GDD_CSV = Path("/content/drive/MyDrive/WERM/outputs/temp/gdd.csv")  # already generated

OUT_CSV  = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_results.csv")

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GDD normalization baseline (keep consistent across runs)
T_BASELINE_GDD = 120.0

# Tillage (GLCM contrast) settings
TILL_FIT_N = 80
TILL_P_LO = 5
TILL_P_HI = 95
TILL_LEVELS = 32
TILL_RESIZE = 256

# Residue damping to reduce straw-edge inflation (keep mild)
TILL_RESIDUE_DAMP_ALPHA = 1.0

# Smooth floor for tillage to avoid hard zeros
TILL_FLOOR = 0.02

# Fusion weights (geometric mean in log space)
WM, WG, WR, WT = 0.30, 0.20, 0.20, 0.30
# ==========================================


# ---------- Moisture reg-only model (matches your ckpt: backbone.* + fc_reg.*) ----------
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


def load_moisture_model(ckpt_path: Path, device: str):
    model = MoistureRegOnly().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Keep only backbone.* and fc_reg.* (drop any cls head if present)
    remapped = {}
    for k, v in sd.items():
        if k.startswith("backbone.") or k.startswith("fc_reg."):
            remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print("Moisture load -> missing:", len(missing), "unexpected:", len(unexpected))
    assert "fc_reg.weight" not in missing and "fc_reg.bias" not in missing, "Moisture fc_reg did not load."
    model.eval()
    return model


# ---------- Residue model ----------
def build_residue_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1), nn.Sigmoid())
    return m


def load_residue_model(ckpt_path: Path, device: str):
    model = build_residue_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Residue load -> missing:", len(missing), "unexpected:", len(unexpected))
    model.eval()
    return model


# ---------- GDD ----------
def load_gdd():
    df = pd.read_csv(GDD_CSV)
    total = float(df["Daily_GDD_Cd"].sum())
    g_norm = min(total / T_BASELINE_GDD, 1.0)
    return g_norm, total


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


def tillage_raw_from_pil(image: Image.Image):
    arr = np.array(image.resize((TILL_RESIZE, TILL_RESIZE)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return glcm_contrast(gray, levels=TILL_LEVELS)


def fit_tillage_norm(img_paths, n_fit=80, p_lo=5, p_hi=95):
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
        return 0.6, 18.0

    lo = float(np.percentile(raws, p_lo))
    hi = float(np.percentile(raws, p_hi))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def norm01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main():
    print("Device:", DEVICE)

    imgs = sorted([p for p in FUSION_IMG_DIR.iterdir()
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print("Images:", len(imgs))

    # Fit tillage normalizer on this batch (for consistent 0..1 scaling on THIS run)
    till_lo, till_hi = fit_tillage_norm(imgs, n_fit=TILL_FIT_N, p_lo=TILL_P_LO, p_hi=TILL_P_HI)
    print("Tillage normalizer fitted. LO:", till_lo, "HI:", till_hi)

    # Load models
    moisture_model = load_moisture_model(MOISTURE_CKPT, DEVICE)
    residue_model = load_residue_model(RESIDUE_CKPT, DEVICE)

    # GDD
    G_norm, G_total = load_gdd()
    print("GDD total:", G_total, "| Normalized GDD:", G_norm)

    rows = []
    eps = 1e-6

    for p in tqdm(imgs, desc="Processing fusion"):
        image = Image.open(p).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            M = float(moisture_model(x).item())   # already 0..1 (relative wetness)
            R = float(residue_model(x).item())    # 0..1 residue fraction

        # Tillage
        T_raw = tillage_raw_from_pil(image)
        T_norm = norm01(T_raw, till_lo, till_hi)

        # Smooth floor + mild gamma (avoid hard zeros, reduce saturation)
        T_norm = TILL_FLOOR + (1.0 - TILL_FLOOR) * (T_norm ** 0.7)

        # Residue damping: straw edges inflate texture => damp tillage when residue high
        T_eff = T_norm * ((1.0 - R) ** TILL_RESIDUE_DAMP_ALPHA)

        # Weighted geometric mean fusion in log space (stable, avoids domination)
        M_term = max(M, eps)
        G_term = max(G_norm, eps)
        R_term = max(1.0 - R, eps)
        T_term = max(T_eff, eps)

        risk_raw = float(np.exp(
            WM * np.log(M_term) +
            WG * np.log(G_term) +
            WR * np.log(R_term) +
            WT * np.log(T_term)
        ))

        rows.append({
            "image": p.name,
            "moisture": M,
            "residue": R,
            "gdd_norm": G_norm,
            "tillage_raw": T_raw,
            "tillage_norm": T_norm,
            "tillage_eff": T_eff,
            "risk_raw": risk_raw
        })

    df = pd.DataFrame(rows)

    # ===== Convert risk_raw -> 0..100 without “all 1s/100s” =====
    # Robust log-space tanh mapping (smooth, no hard clipping)
    eps2 = 1e-9
    log_r = np.log(df["risk_raw"].values + eps2)

    mu = float(np.median(log_r))
    mad = float(np.median(np.abs(log_r - mu)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(log_r) + 1e-6)

    z = (log_r - mu) / sigma
    score_0_100 = 50.0 + 50.0 * np.tanh(z / 2.0)

    df["risk_score_0_100"] = score_0_100

    # ===== A–E classification =====
    bins = [-1e9, 20, 40, 60, 80, 1e9]
    labels = ["A", "B", "C", "D", "E"]
    df["risk_class"] = pd.cut(df["risk_score_0_100"], bins=bins, labels=labels, right=False)

    # Sort by score
    df = df.sort_values("risk_score_0_100", ascending=False)

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Print bin counts
    counts = df["risk_class"].value_counts().reindex(labels).fillna(0).astype(int)
    print("\n=== A–E counts ===")
    for k in labels:
        print(f"{k}: {counts[k]}")

    print("\nScore range:", float(df["risk_score_0_100"].min()), "to", float(df["risk_score_0_100"].max()))
    print("\nSaved:", OUT_CSV)

    print("\nTop 10:")
    print(df.head(10)[["image","risk_class","risk_score_0_100","moisture","residue","tillage_eff","gdd_norm"]])

    print("\nBottom 10:")
    print(df.tail(10)[["image","risk_class","risk_score_0_100","moisture","residue","tillage_eff","gdd_norm"]])

    # ===== Sensitivity + contribution analysis (log-space) =====
    # For risk_raw = exp( sum_i w_i log(x_i) )
    # Elasticity of risk_raw wrt x_i is constant: d ln(risk)/d ln(x_i) = w_i
    # But per-image "contribution" differs because log(x_i) differs.

    SENS_CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_sensitivity.csv")
    SUM_CSV  = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_global_summary.csv")

    eps_s = 1e-6

    # terms (clamped to avoid log(0))
    df["M_term"] = np.clip(df["moisture"].values, eps_s, 1.0)
    df["G_term"] = np.clip(df["gdd_norm"].values, eps_s, 1.0)
    df["R_term"] = np.clip((1.0 - df["residue"].values), eps_s, 1.0)
    df["T_term"] = np.clip(df["tillage_eff"].values, eps_s, 1.0)

    # log terms
    df["logM"] = np.log(df["M_term"])
    df["logG"] = np.log(df["G_term"])
    df["logR"] = np.log(df["R_term"])
    df["logT"] = np.log(df["T_term"])

    # weighted log contributions (these sum to log(risk_raw))
    df["c_logM"] = WM * df["logM"]
    df["c_logG"] = WG * df["logG"]
    df["c_logR"] = WR * df["logR"]
    df["c_logT"] = WT * df["logT"]
    df["log_risk_raw_recon"] = df["c_logM"] + df["c_logG"] + df["c_logR"] + df["c_logT"]

    # sanity recon (optional)
    df["risk_raw_recon"] = np.exp(df["log_risk_raw_recon"])

    # per-image normalized contribution shares in log-space (absolute)
    # tells “which factor drove log-risk magnitude” on that image
    abs_sum = (df["c_logM"].abs() + df["c_logG"].abs() + df["c_logR"].abs() + df["c_logT"].abs()).values + 1e-12
    df["share_logM"] = (df["c_logM"].abs().values / abs_sum)
    df["share_logG"] = (df["c_logG"].abs().values / abs_sum)
    df["share_logR"] = (df["c_logR"].abs().values / abs_sum)
    df["share_logT"] = (df["c_logT"].abs().values / abs_sum)

    # Elasticities (constant for multiplicative model)
    df["elas_M"] = WM
    df["elas_G"] = WG
    df["elas_Rterm"] = WR   # this is elasticity wrt (1-Residue), not Residue itself
    df["elas_T"] = WT

    # Partial derivatives of risk_raw wrt each term (per-image)
    # d risk / d x_i = (w_i * risk) / x_i
    df["dRisk_dM"] = (WM * df["risk_raw"].values) / df["M_term"].values
    df["dRisk_dG"] = (WG * df["risk_raw"].values) / df["G_term"].values
    df["dRisk_dRterm"] = (WR * df["risk_raw"].values) / df["R_term"].values
    df["dRisk_dT"] = (WT * df["risk_raw"].values) / df["T_term"].values

    # Global variance contributions (in log space) across the dataset:
    # Var(log risk) ~ Var(sum c_i). We report Var(c_i) and Corr(c_i, log risk).
    logrisk = df["log_risk_raw_recon"].values

    def _summ(name, arr):
        arr = np.asarray(arr, dtype=float)
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_var": float(np.var(arr)),
            f"{name}_corr_with_logrisk": float(np.corrcoef(arr, logrisk)[0,1]) if np.std(arr) > 1e-12 else 0.0,
        }

    summary = {}
    summary.update(_summ("c_logM", df["c_logM"].values))
    summary.update(_summ("c_logG", df["c_logG"].values))
    summary.update(_summ("c_logR", df["c_logR"].values))
    summary.update(_summ("c_logT", df["c_logT"].values))
    summary["logrisk_std"] = float(np.std(logrisk))
    summary["logrisk_var"] = float(np.var(logrisk))
    summary["n_images"] = int(len(df))

    # Mean “share” per factor
    summary["mean_share_logM"] = float(df["share_logM"].mean())
    summary["mean_share_logG"] = float(df["share_logG"].mean())
    summary["mean_share_logR"] = float(df["share_logR"].mean())
    summary["mean_share_logT"] = float(df["share_logT"].mean())

    # Mean abs partial derivative magnitudes (scale-dependent; useful for ranking sensitivity at current operating point)
    summary["mean_abs_dRisk_dM"] = float(np.mean(np.abs(df["dRisk_dM"].values)))
    summary["mean_abs_dRisk_dG"] = float(np.mean(np.abs(df["dRisk_dG"].values)))
    summary["mean_abs_dRisk_dRterm"] = float(np.mean(np.abs(df["dRisk_dRterm"].values)))
    summary["mean_abs_dRisk_dT"] = float(np.mean(np.abs(df["dRisk_dT"].values)))

    # Save per-image sensitivity table
    SENS_CSV.parent.mkdir(parents=True, exist_ok=True)
    sens_cols = [
        "image",
        "risk_raw","risk_score_0_100","risk_class",
        "moisture","residue","gdd_norm","tillage_eff",
        "c_logM","c_logG","c_logR","c_logT","log_risk_raw_recon",
        "share_logM","share_logG","share_logR","share_logT",
        "elas_M","elas_G","elas_Rterm","elas_T",
        "dRisk_dM","dRisk_dG","dRisk_dRterm","dRisk_dT",
    ]
    df[sens_cols].to_csv(SENS_CSV, index=False)

    # Save global summary
    pd.DataFrame([summary]).to_csv(SUM_CSV, index=False)

    print("\nSaved per-image sensitivity:", SENS_CSV)
    print("Saved global summary:", SUM_CSV)
    print("\n=== Global log-space variance (which factors drive spread) ===")
    print("Var(log risk):", summary["logrisk_var"])
    print("Var(c_logM):", summary["c_logM_var"], "| Corr:", summary["c_logM_corr_with_logrisk"])
    print("Var(c_logR):", summary["c_logR_var"], "| Corr:", summary["c_logR_corr_with_logrisk"])
    print("Var(c_logT):", summary["c_logT_var"], "| Corr:", summary["c_logT_corr_with_logrisk"])
    print("Var(c_logG):", summary["c_logG_var"], "| Corr:", summary["c_logG_corr_with_logrisk"], "(should be ~0 if G constant)")

if __name__ == "__main__":
    main()