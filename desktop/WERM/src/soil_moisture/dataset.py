import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def build_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tfms, val_tfms

class SoilMoistureFolderDataset(Dataset):
    """
    Expects:
      root/Dry/*.jpg
      root/Moderate/*.jpg
      root/Wet/*.jpg
    """
    def __init__(self, root: Path, label_map: dict, transform):
        self.root = Path(root)
        self.label_map = label_map
        self.transform = transform
        self.samples = []

        for cls, idx in label_map.items():
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")

            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append((str(cls_dir / fn), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root} with extensions {IMG_EXTS}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long), path
