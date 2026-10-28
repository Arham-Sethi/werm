import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report

from WERM.src.soil_moisture.config import CFG
from WERM.src.soil_moisture.dataset import SoilMoistureFolderDataset, build_transforms
from WERM.src.soil_moisture.utils import device, save_json

def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--out", type=str, default=None, help="Optional output path for eval.json")
    args = parser.parse_args()

    cfg = CFG()
    dev = device()

    # Use val transforms (no augmentation)
    _train_tfms, val_tfms = build_transforms(cfg.img_size)

    from torch.utils.data import Subset  # add at top

    # Build full dataset (no augmentation)
    ds_all = SoilMoistureFolderDataset(cfg.data_root, cfg.label_map, transform=val_tfms)

    model = build_model(3).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Use the SAME val split used during training
    val_idxs = None
    if isinstance(ckpt, dict):
        meta = ckpt.get("meta", {})
        val_idxs = meta.get("val_indices", None)

    if val_idxs is None:
        raise ValueError(
            "Checkpoint does not contain val_indices. "
            "Re-train after adding val_indices/train_indices to meta in train.py."
        )

    ds = Subset(ds_all, val_idxs)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


    y_true, y_pred = [], []
    with torch.no_grad():
        for x, yb, _paths in loader:
            x = x.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=[cfg.label_names[i] for i in range(3)],
        output_dict=True
    )

    result = {
        "ckpt": args.ckpt,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_names": cfg.label_names,
    }

    out_path = Path(args.out) if args.out else (Path(args.ckpt).parent / "eval.json")
    save_json(result, out_path)
    print("Saved eval to:", out_path)
    print("\nConfusion matrix:\n", cm)

if __name__ == "__main__":
    main()
