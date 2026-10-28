# WERM/src/soil_moisture/infer_folder.py

import argparse
from pathlib import Path
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_model(num_classes=3):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_val_tfms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class FolderDataset(Dataset):
    """
    Expects any subset of:
      root/Dry
      root/Moderate
      root/Wet

    Works even if Moderate is missing.
    """
    def __init__(self, root: Path, class_to_idx: dict, transform):
        self.samples = []
        self.transform = transform

        for cls, idx in class_to_idx.items():
            cls_dir = root / cls
            if not cls_dir.exists():
                continue  # allow missing classes (e.g., no Moderate)

            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append((str(cls_dir / fn), idx, cls))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y, cls_name = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long), path, cls_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Path to trained best.pt")
    ap.add_argument("--data_root", required=True, type=str, help="Folder with Dry/Wet[/Moderate]")
    ap.add_argument("--out_csv", default=None, type=str, help="Optional output CSV path")
    ap.add_argument("--img_size", default=224, type=int)
    args = ap.parse_args()

    class_to_idx = {"Dry": 0, "Moderate": 1, "Wet": 2}
    idx_to_name = {0: "Dry", 1: "Moderate", 2: "Wet"}

    dev = get_device()
    tfm = build_val_tfms(args.img_size)

    ds = FolderDataset(Path(args.data_root), class_to_idx, tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(3).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_csv = args.out_csv or (Path(args.ckpt).parent / "unseen_preds.csv")
    out_csv = Path(out_csv)

    correct = 0
    total = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "true_label", "pred_label",
            "p_dry", "p_moderate", "p_wet", "correct"
        ])

        with torch.no_grad():
            for x, y, paths, true_names in loader:
                x = x.to(dev)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                y_np = y.numpy()

                for i in range(len(paths)):
                    pred_name = idx_to_name[int(preds[i])]
                    ok = int(preds[i] == y_np[i])
                    correct += ok
                    total += 1

                    writer.writerow([
                        paths[i],
                        true_names[i],
                        pred_name,
                        probs[i][0],
                        probs[i][1],
                        probs[i][2],
                        ok
                    ])

    acc = correct / max(1, total)
    print(f"Saved predictions to: {out_csv}")
    print(f"Unseen Dry/Wet accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
