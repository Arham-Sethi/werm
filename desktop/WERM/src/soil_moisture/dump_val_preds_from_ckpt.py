from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

from train import build_model
from dataset import SoilMoistureFolderDataset, build_transforms
from utils import device

CKPT_PATH = Path("/content/drive/MyDrive/WERM/outputs/runs/AA_70_30_ep30_20260127_044224/best.pt")

def main():
    dev = device()

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    meta = ckpt["meta"]

    data_root = Path(meta["data_root"])
    img_size = meta["img_size"]
    label_map = meta["label_map"]
    val_indices = meta["val_indices"]

    # EXACT eval transforms used in training
    _, val_tfms = build_transforms(img_size)

    full_ds = SoilMoistureFolderDataset(data_root, label_map, transform=val_tfms)
    val_ds = Subset(full_ds, val_indices)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=3)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    rows = []
    with torch.no_grad():
        for x, y, paths in val_loader:
            x = x.to(dev, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred = probs.argmax(dim=1)

            W = 0.0*probs[:,0] + 0.5*probs[:,1] + 1.0*probs[:,2]
            conf, _ = probs.max(dim=1)
            eps = 1e-12
            ent = -(probs * (probs + eps).log()).sum(dim=1)

            probs_cpu = probs.detach().cpu().numpy()
            pred_cpu = pred.detach().cpu().numpy()
            W_cpu = W.detach().cpu().numpy()
            conf_cpu = conf.detach().cpu().numpy()
            ent_cpu = ent.detach().cpu().numpy()

            y_cpu = y.numpy()
            paths = list(paths)

            for i in range(len(paths)):
                rows.append({
                    "path": paths[i],
                    "gt": int(y_cpu[i]),
                    "pred": int(pred_cpu[i]),
                    "p0": float(probs_cpu[i,0]),
                    "p1": float(probs_cpu[i,1]),
                    "p2": float(probs_cpu[i,2]),
                    "W": float(W_cpu[i]),
                    "confidence": float(conf_cpu[i]),
                    "entropy": float(ent_cpu[i]),
                })

    df = pd.DataFrame(rows)
    out_csv = CKPT_PATH.parent / "val_preds_best.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
