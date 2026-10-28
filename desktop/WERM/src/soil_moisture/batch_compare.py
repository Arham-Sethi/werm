# WERM/src/soil_moisture/batch_compare.py
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import pandas as pd

IMG_BATCH = 64


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_val_dataset(data_root: Path, img_size: int, label_map: dict):
    # Local minimal dataset implementation to avoid import/path issues
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    import os

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    class FolderDS(Dataset):
        def __init__(self, root: Path, label_map: dict):
            self.root = Path(root)
            self.samples = []
            for cls, idx in label_map.items():
                cls_dir = self.root / cls
                if not cls_dir.exists():
                    raise FileNotFoundError(f"Missing class folder: {cls_dir}")
                for fn in os.listdir(cls_dir):
                    if fn.lower().endswith(IMG_EXTS):
                        self.samples.append((str(cls_dir / fn), int(idx)))

            if len(self.samples) == 0:
                raise RuntimeError(f"No images found under {root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, y = self.samples[i]
            img = Image.open(path).convert("RGB")
            img = val_tfms(img)
            return img, torch.tensor(y, dtype=torch.long), path

    return FolderDS(data_root, label_map)


def build_softmax_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_feature_extractor():
    m = models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m


def load_state_no_fc(feature_model: nn.Module, state: dict):
    filtered = {k: v for k, v in state.items() if not k.startswith("fc.")}
    feature_model.load_state_dict(filtered, strict=False)


@torch.no_grad()
def eval_softmax_on_val(model, loader, dev, label_names):
    model.eval()
    y_true, y_pred = [], []
    for x, yb, _ in loader:
        x = x.to(dev)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(yb.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return acc, cm, report


@torch.no_grad()
def extract_features(model, loader, dev):
    model.eval()
    X_list, y_list = [], []
    for x, yb, _ in loader:
        x = x.to(dev)
        feats = model(x).cpu().numpy()
        X_list.append(feats)
        y_list.append(yb.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _label_names_from_meta(meta):
    ln = meta.get("label_names", None)
    # Your saved meta might be dict with int keys, or str keys depending on json dump.
    if isinstance(ln, dict):
        names = []
        for i in range(3):
            if i in ln:
                names.append(str(ln[i]))
            elif str(i) in ln:
                names.append(str(ln[str(i)]))
            else:
                names.append(str(i))
        return names
    # fallback
    return ["0", "1", "2"]


def run_one(run_dir: Path, ckpt_name: str, svm_kernel: str, svm_C: float, svm_gamma: str):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dev = device()
    ckpt = torch.load(str(ckpt_path), map_location=dev)

    if not isinstance(ckpt, dict) or "model" not in ckpt or "meta" not in ckpt:
        raise ValueError(f"Checkpoint format unexpected in {ckpt_path}")

    meta = ckpt["meta"]
    state = ckpt["model"]

    data_root = Path(meta["data_root"])
    img_size = int(meta.get("img_size", 224))
    label_map = meta.get("label_map", {"Dry": 0, "Moderate": 1, "Wet": 2})
    label_names = _label_names_from_meta(meta)

    train_idxs = meta.get("train_indices", None)
    val_idxs = meta.get("val_indices", None)
    if train_idxs is None or val_idxs is None:
        raise ValueError(
            f"{ckpt_path} missing train_indices/val_indices in ckpt['meta'].\n"
            f"Fix: add them in train.py meta and retrain for this run."
        )

    ds_all = build_val_dataset(data_root, img_size, label_map)
    train_ds = Subset(ds_all, train_idxs)
    val_ds = Subset(ds_all, val_idxs)

    train_loader = DataLoader(train_ds, batch_size=IMG_BATCH, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=IMG_BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # Softmax eval
    softmax_model = build_softmax_model(3).to(dev)
    softmax_model.load_state_dict(state)
    softmax_acc, softmax_cm, softmax_rep = eval_softmax_on_val(softmax_model, val_loader, dev, label_names)

    # SVM eval
    feat_model = build_feature_extractor().to(dev)
    load_state_no_fc(feat_model, state)

    X_train, y_train = extract_features(feat_model, train_loader, dev)
    X_val, y_val = extract_features(feat_model, val_loader, dev)

    gamma = svm_gamma
    try:
        gamma = float(gamma)
    except Exception:
        pass

    svm = SVC(kernel=svm_kernel, C=svm_C, gamma=gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)

    svm_acc = accuracy_score(y_val, y_pred)
    svm_cm = confusion_matrix(y_val, y_pred)
    svm_rep = classification_report(y_val, y_pred, target_names=label_names, output_dict=True)

    # Save per-run json
    out = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "softmax_val_acc": float(softmax_acc),
        "svm_val_acc": float(svm_acc),
        "softmax_confusion_matrix": softmax_cm.tolist(),
        "svm_confusion_matrix": svm_cm.tolist(),
        "softmax_report": softmax_rep,
        "svm_report": svm_rep,
        "svm_params": {"kernel": svm_kernel, "C": svm_C, "gamma": str(svm_gamma)},
        "meta": {
            "train_frac": meta.get("train_frac", None),
            "epochs": meta.get("epochs", None),
            "seed": meta.get("seed", None),
            "img_size": img_size,
            "data_root": str(data_root),
        },
    }
    with open(run_dir / "compare_softmax_vs_svm.json", "w") as f:
        json.dump(out, f, indent=2)

    return {
        "run_folder": run_dir.name,
        "train_frac": meta.get("train_frac", None),
        "epochs": meta.get("epochs", None),
        "softmax_val_acc": float(softmax_acc),
        "svm_val_acc": float(svm_acc),
        "ckpt": ckpt_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Pass 3 folders (or any number)")
    ap.add_argument("--ckpt", default="best.pt", choices=["best.pt", "last.pt"])
    ap.add_argument("--svm_kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    ap.add_argument("--svm_C", type=float, default=10.0)
    ap.add_argument("--svm_gamma", default="scale", help="'scale'/'auto' or float string")
    ap.add_argument("--save_csv", default=None, help="Optional CSV path")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        rows.append(run_one(Path(r), args.ckpt, args.svm_kernel, args.svm_C, args.svm_gamma))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["train_frac", "epochs", "run_folder"], ascending=[True, True, True])

    print("\n=== Softmax vs SVM (val split) ===\n")
    print(df.to_string(index=False))

    out_csv = Path(args.save_csv) if args.save_csv else (Path(args.runs[0]).parent / "compare_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\nSaved table CSV to:", out_csv)


if __name__ == "__main__":
    main()
