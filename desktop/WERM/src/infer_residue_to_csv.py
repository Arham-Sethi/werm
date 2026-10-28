from pathlib import Path
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)  # regression head
    return m

def list_images(root: Path):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix in exts])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True, help="Folder containing images (recursive)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint")
    ap.add_argument("--out_csv", type=str, required=True, help="Where to save residue_preds.csv")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--use_sigmoid", action="store_true", help="Apply sigmoid to output (if model trained bounded)")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    ckpt_path = Path(args.ckpt)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    assert img_root.exists(), f"img_root not found: {img_root}"
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dev).eval()

    paths = list_images(img_root)
    print("Images found:", len(paths))
    if len(paths) == 0:
        raise RuntimeError("No images found under img_root.")

    rows = []

    def load_one(p: Path):
        img = Image.open(p).convert("RGB")
        return tfm(img)

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch), desc="Residue inference"):
            batch_paths = paths[i:i+args.batch]
            xs = []
            good_paths = []
            for p in batch_paths:
                try:
                    xs.append(load_one(p))
                    good_paths.append(p)
                except Exception as e:
                    print("Skipping bad image:", p.name, e)
            if len(xs) == 0:
                continue

            x = torch.stack(xs, dim=0).to(dev)
            y = model(x).squeeze(1)  # shape [B]

            if args.use_sigmoid:
                y = torch.sigmoid(y)

            pred = y.detach().cpu().numpy().astype(np.float32)
            pred = np.clip(pred, 0.0, 1.0)

            for pth, pr in zip(good_paths, pred):
                rows.append({
                    "sample_id": pth.name,
                    "rel_path": str(pth.relative_to(img_root)),
                    "pred_fR": float(pr),
                    "R_residue": float(1.0 - pr)
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
