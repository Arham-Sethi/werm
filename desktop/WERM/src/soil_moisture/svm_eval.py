# WERM/src/soil_moisture/svm_eval.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

from WERM.src.soil_moisture.config import CFG
from WERM.src.soil_moisture.dataset import SoilMoistureFolderDataset, build_transforms
from WERM.src.soil_moisture.utils import device, save_json


def build_feature_extractor():
    # Same backbone as training, but remove classifier head
    m = models.resnet18(weights=None)
    m.fc = nn.Identity()  # outputs 512-d features
    return m


def load_backbone_weights(feature_model: nn.Module, ckpt_state: dict):
    """
    Your training checkpoint was saved from a ResNet18 with m.fc = Linear(..., 3).
    We now have m.fc = Identity(), so we load everything except the fc weights.
    """
    state = ckpt_state["model"] if "model" in ckpt_state else ckpt_state

    filtered = {}
    for k, v in state.items():
        if k.startswith("fc."):
            continue
        filtered[k] = v

    missing, unexpected = feature_model.load_state_dict(filtered, strict=False)
    return missing, unexpected


@torch.no_grad()
def extract_features(model, loader, dev):
    model.eval()
    X_list, y_list = [], []

    for x, yb, _paths in loader:
        x = x.to(dev, non_blocking=True)
        feats = model(x)  # [B, 512]
        X_list.append(feats.cpu().numpy())
        y_list.append(yb.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output dir (default: ckpt folder)")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=str, default="scale", help="For rbf/poly/sigmoid: 'scale' or 'auto' or float-like string")
    args = parser.parse_args()

    cfg = CFG()
    dev = device()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location=dev)

    # Transforms: no augmentation for feature extraction/eval
    _train_tfms, val_tfms = build_transforms(cfg.img_size)

    # Full dataset
    ds_all = SoilMoistureFolderDataset(cfg.data_root, cfg.label_map, transform=val_tfms)

    # Use SAME split indices saved by your train.py (recommended)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    train_idxs = meta.get("train_indices", None)
    val_idxs = meta.get("val_indices", None)

    if train_idxs is None or val_idxs is None:
        raise ValueError(
            "Checkpoint does not contain train_indices/val_indices.\n"
            "Fix: in train.py add them to meta, retrain once, then rerun svm_eval.py."
        )

    train_ds = Subset(ds_all, train_idxs)
    val_ds = Subset(ds_all, val_idxs)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Feature extractor
    feat_model = build_feature_extractor().to(dev)
    missing, unexpected = load_backbone_weights(feat_model, ckpt)
    # (missing may include 'fc.weight'/'fc.bias' depending on how state_dict saved; that's OK)

    # Extract features
    X_train, y_train = extract_features(feat_model, train_loader, dev)
    X_val, y_val = extract_features(feat_model, val_loader, dev)

    # Gamma parsing: allow float string
    gamma = args.gamma
    try:
        gamma = float(gamma)
    except Exception:
        pass  # keep 'scale'/'auto'

    # Train SVM on deep features
    svm = SVC(kernel=args.kernel, C=args.C, gamma=gamma)
    svm.fit(X_train, y_train)

    # Evaluate
    y_pred = svm.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(
        y_val, y_pred,
        target_names=[cfg.label_names[i] for i in range(3)],
        output_dict=True
    )

    # Save outputs
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = out_dir / "svm.joblib"
    joblib.dump(svm, joblib_path)

    result = {
        "ckpt": str(ckpt_path),
        "svm": {"kernel": args.kernel, "C": args.C, "gamma": str(args.gamma)},
        "val_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_names": cfg.label_names,
        "notes": "SVM trained on ResNet18 deep features (fc removed)."
    }
    save_json(result, out_dir / "svm_eval.json")

    print("SVM eval saved to:", out_dir / "svm_eval.json")
    print("SVM model saved to:", joblib_path)
    print("\nVal accuracy:", acc)
    print("\nConfusion matrix:\n", cm)


if __name__ == "__main__":
    main()
