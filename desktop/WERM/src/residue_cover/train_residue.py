from pathlib import Path
import json
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset_residue import ResidueDataset

# ---------------- CONFIG ----------------
DATA_ROOT = Path("/content/drive/MyDrive/WERM/data/residue_usgs")

TRAIN_CSV = DATA_ROOT / "train_strat.csv"
VAL_CSV   = DATA_ROOT / "val_strat.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 25
WEIGHT_DECAY = 1e-2

RUN_NAME = "residue_r18_strat_logcosh"
OUT_ROOT = Path("/content/drive/MyDrive/WERM/outputs/runs")
# ----------------------------------------

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 1),
        nn.Sigmoid()
    )
    return m

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    mae_sum = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(dev)
        y = y.to(dev).unsqueeze(1)

        pred = model(x)
        mae_sum += (pred - y).abs().sum().item()
        n += y.numel()

    return mae_sum / max(1, n)

def main():
    dev = device()

    run_dir = OUT_ROOT / f"{RUN_NAME}_{timestamp()}"
    ensure_dir(run_dir)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(0.25, 0.25, 0.15, 0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                         # MUST come before RandomErasing
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225]),
    ])


    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    train_ds = ResidueDataset(DATA_ROOT, TRAIN_CSV, transform=train_tfms)
    val_ds   = ResidueDataset(DATA_ROOT, VAL_CSV, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model().to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val_mae = 1e9

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(dev)
            y = y.to(dev).unsqueeze(1)

            pred = model(x)
            loss = log_cosh_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step()

        val_mae = evaluate(model, val_loader, dev)
        writer.add_scalar("val/mae", val_mae, epoch)

        print(f"Epoch {epoch}: val_mae = {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"model": model.state_dict()}, run_dir / "best.pt")

    writer.close()
    print("Best MAE:", best_val_mae)

if __name__ == "__main__":
    main()
