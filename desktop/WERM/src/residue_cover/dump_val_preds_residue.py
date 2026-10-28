from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_residue import ResidueDataset

CKPT = Path("/content/drive/MyDrive/WERM/outputs/runs/residue_r18_sigmoid_mae_yearsplit_20260211_172736/best.pt")
DATA_ROOT = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
CSV_PATH = DATA_ROOT / "labels.csv"

IMG_SIZE = 224
VAL_YEARS = [2021, 2022]

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    from torchvision import models
    import torch.nn as nn
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1), nn.Sigmoid())
    return m

def main():
    dev = device()
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = build_model()
    model.load_state_dict(state)
    model.to(dev).eval()

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    ds = ResidueDataset(DATA_ROOT, CSV_PATH, years=VAL_YEARS, transform=val_tfms)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    rows = []
    with torch.no_grad():
        for x, y, names in loader:
            x = x.to(dev)
            pred = model(x).squeeze(1).detach().cpu().numpy()
            y = y.numpy()
            for i in range(len(names)):
                rows.append({
                    "photo_id": names[i],
                    "gt_fR": float(y[i]),
                    "pred_fR": float(pred[i]),
                    "abs_err": float(abs(pred[i] - y[i])),
                })

    out_csv = CKPT.parent / "val_preds_best.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv, "rows:", len(rows))

if __name__ == "__main__":
    main()
