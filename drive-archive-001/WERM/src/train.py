from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter  # <-- TensorBoard
from dataset import SoilMoistureFolderDataset, build_transforms
from utils import set_seed, ensure_dir, timestamp, save_json, device
import argparse


def build_model(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config", choices=["config", "config2"])
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root folder")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--train_frac", type=float, default=None, help="Override train split fraction (e.g. 0.7)")
    args = parser.parse_args()

    # choose which CFG to import
    if args.config == "config2":
        from config2 import CFG as CFG_SELECTED
    else:
        from config import CFG as CFG_SELECTED

    cfg = CFG_SELECTED()

    # overrides
    if args.run_name:
        cfg.run_name = args.run_name
    if args.data_root:
        from pathlib import Path
        cfg.data_root = Path(args.data_root)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.train_frac is not None:
        cfg.train_frac = float(args.train_frac)



    set_seed(cfg.seed)
    dev = device()

    # Basic sanity checks (fail fast)
    if not cfg.project_root.exists():
        raise FileNotFoundError(f"Project root not found: {cfg.project_root}")
    if not cfg.data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {cfg.data_root}\n"
            f"Expected: /content/drive/MyDrive/WERM/data/soil_moisture/Dry|Moderate|Wet"
        )

    run_dir = cfg.out_root / f"{cfg.run_name}_{timestamp()}"
    ensure_dir(run_dir)

    # TensorBoard log dir (saved on Drive)
    tb_dir = run_dir / "tb"
    ensure_dir(tb_dir)
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_tfms, val_tfms = build_transforms(cfg.img_size)

    # Full dataset for split labels
    full_ds_for_split = SoilMoistureFolderDataset(cfg.data_root, cfg.label_map, transform=val_tfms)
    y = np.array([full_ds_for_split.samples[i][1] for i in range(len(full_ds_for_split))])
    idxs = np.arange(len(full_ds_for_split))

    if cfg.stratified:
        train_idxs, val_idxs = train_test_split(
            idxs, train_size=cfg.train_frac, random_state=cfg.seed, stratify=y
        )
    else:
        train_idxs, val_idxs = train_test_split(
            idxs, train_size=cfg.train_frac, random_state=cfg.seed
        )
    train_idxs_list = train_idxs.tolist()
    val_idxs_list = val_idxs.tolist()
    
    # Datasets with different transforms
    train_ds = SoilMoistureFolderDataset(cfg.data_root, cfg.label_map, transform=train_tfms)
    val_ds   = SoilMoistureFolderDataset(cfg.data_root, cfg.label_map, transform=val_tfms)

    train_ds = Subset(train_ds, train_idxs_list)
    val_ds   = Subset(val_ds, val_idxs_list)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = build_model(num_classes=3).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    meta = {
        "project_root": str(cfg.project_root),
        "data_root": str(cfg.data_root),
        "out_root": str(cfg.out_root),
        "label_map": cfg.label_map,
        "label_names": cfg.label_names,
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "train_frac": cfg.train_frac,
        "stratified": cfg.stratified,
        "seed": cfg.seed,
        "device": dev,
        "run_dir": str(run_dir),
        "tb_dir": str(tb_dir),
        "model": "resnet18",
        "task": "moisture_emergence_permissivity_proxy",
        "train_indices": train_idxs_list,
        "val_indices": val_idxs_list,
    }
    save_json(meta, run_dir / "meta.json")

    # Log hyperparams to TensorBoard (as text)
    writer.add_text("meta/run_dir", str(run_dir))
    writer.add_text("meta/data_root", str(cfg.data_root))
    writer.add_text("meta/labels", str(cfg.label_names))
    writer.add_text("meta/hparams", str({k: meta[k] for k in ["img_size", "batch_size", "epochs", "lr", "weight_decay", "train_frac", "seed"]}))

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, yb, _paths in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]"):
            x = x.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # batch metrics
            bs = x.size(0)
            train_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += bs

            # TensorBoard: batch loss
            writer.add_scalar("train/batch_loss", float(loss.item()), global_step)
            global_step += 1

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, yb, _paths in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                x = x.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, yb)

                bs = x.size(0)
                val_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += bs

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        save_json(history, run_dir / "history.json")

        # TensorBoard: epoch metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        # Save checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "meta": meta}, last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "meta": meta}, best_path)

        print(f"\nEpoch {epoch:02d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} (best={best_val_acc:.3f})\n")

    writer.close()

    print(f"Done.\nRun folder: {run_dir}\nTensorBoard: {tb_dir}\nBest checkpoint: {best_path}")

if __name__ == "__main__":
    main()
