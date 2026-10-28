from pathlib import Path
import torch
import math
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn

from dataset_residue import ResidueDataset

DATA_ROOT = Path("/content/drive/MyDrive/WERM/data/residue_usgs")
TEST_CSV = DATA_ROOT / "test_strat.csv"

CKPT_PATH = Path("/content/drive/MyDrive/WERM/outputs/runs/residue_r18_strat_logcosh_20260214_142945/best.pt")

IMG_SIZE = 224
BATCH_SIZE = 32


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(dev)
        y = y.to(dev).unsqueeze(1)

        pred = model(x)
        err = pred - y

        mae_sum += err.abs().sum().item()
        mse_sum += (err * err).sum().item()
        n += y.numel()

    mae = mae_sum / max(1, n)
    rmse = math.sqrt(mse_sum / max(1, n))

    return mae, rmse


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 1),
        nn.Sigmoid()
    )
    return m


def main():
    dev = device()
    print("Device:", dev)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    test_ds = ResidueDataset(DATA_ROOT, TEST_CSV, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Test samples:", len(test_ds))

    model = build_model().to(dev)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    test_mae, test_rmse = evaluate(model, test_loader, dev)

    print("\n===== TEST RESULTS =====")
    print("MAE (fraction):", test_mae)
    print("MAE (%):", test_mae * 100.0)
    print("RMSE (fraction):", test_rmse)
    print("RMSE (%):", test_rmse * 100.0)


if __name__ == "__main__":
    main()
