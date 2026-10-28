from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from train import build_model          # same model builder
from dataset import build_transforms   # SAME transforms as training
from utils import device

# ---------------- CONFIG ----------------
CKPT_PATH = Path("/content/drive/MyDrive/WERM/outputs/runs/AA_70_30_ep30_20260127_044224/best.pt")
DATA_ROOT = Path("/content/drive/MyDrive/WERM/data/unseen_moisture_mini")
BATCH_SIZE = 32
# ---------------------------------------

# IMPORTANT: must match training label order
# Check your cfg.label_map — assumed here:
IDX_DRY = 0
IDX_MOD = 1
IDX_WET = 2


def main():
    dev = device()

    # ---------- Load checkpoint ----------
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    meta = ckpt["meta"]

    img_size = meta["img_size"]
    label_names = meta["label_names"]

    print("Loaded checkpoint trained on labels:", label_names)
    print("Image size:", img_size)

    # ---------- Build model ----------
    model = build_model(num_classes=3)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    # ---------- Build EXACT eval transforms ----------
    _, val_tfms = build_transforms(img_size)

    # ---------- Dataset ----------
    ds = datasets.ImageFolder(DATA_ROOT, transform=val_tfms)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Unseen dataset classes (from folders):", ds.classes)
    class_to_idx = ds.class_to_idx

    assert "Dry" in class_to_idx and "Wet" in class_to_idx, \
        f"Expected Dry/Wet folders, got {class_to_idx}"

    total = 0
    correct_binary = 0
    correct_strict = 0

    results = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            p_dry = probs[:, IDX_DRY]
            p_mod = probs[:, IDX_MOD]
            p_wet = probs[:, IDX_WET]

            # ---------- Continuous wetness ----------
            W = 0.0 * p_dry + 0.5 * p_mod + 1.0 * p_wet

            # Binary prediction via W
            pred_bin = (W >= 0.5).long()   # 1=Wet, 0=Dry

            # Strict top-1 prediction
            pred_top = probs.argmax(dim=1)
            pred_strict = torch.full_like(pred_bin, -1)  # stays on same device as pred_bin
            pred_strict[pred_top == IDX_DRY] = 0
            pred_strict[pred_top == IDX_WET] = 1

            # Ground truth from folders (keep on CPU for easy metrics)
            y_bin = torch.zeros_like(y)                 # y is CPU
            y_bin[y == class_to_idx["Wet"]] = 1

            total += y.size(0)

            pred_bin_cpu = pred_bin.cpu()
            pred_strict_cpu = pred_strict.cpu()

            correct_binary += (pred_bin_cpu == y_bin).sum().item()
            correct_strict += ((pred_strict_cpu == y_bin) & (pred_strict_cpu != -1)).sum().item()

            for i in range(y.size(0)):
                results.append({
                    "gt": int(y_bin[i]),
                    "W": float(W[i]),
                    "p_dry": float(p_dry[i]),
                    "p_mod": float(p_mod[i]),
                    "p_wet": float(p_wet[i]),
                    "pred_bin": int(pred_bin[i]),
                    "pred_strict": int(pred_strict[i]),
                })

    acc_bin = correct_binary / total
    acc_strict = correct_strict / total

    report = {
        "total_images": total,
        "binary_acc_via_W>=0.5": acc_bin,
        "strict_top1_acc_(Moderate_penalized)": acc_strict,
        "note": "Binary uses continuous wetness; strict penalizes Moderate predictions",
    }

    out_path = Path("/content/drive/MyDrive/WERM/outputs/unseen_moisture_mini_eval.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "samples": results}, f, indent=2)

    print(json.dumps(report, indent=2))
    print("Saved detailed results to:", out_path)


if __name__ == "__main__":
    main()
