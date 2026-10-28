import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class INSAMoistureDataset(Dataset):
    """
    Reads labels from CSV with columns: image_path, moisture
    Crops top strip to remove the visible "Water content XX%" tag.
    """
    def __init__(self, csv_path, transform=None, normalize_to_01=False, crop_top_frac=0.18):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.normalize_to_01 = normalize_to_01
        self.crop_top_frac = float(crop_top_frac)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        moisture = float(row["moisture"])

        if self.normalize_to_01:
            moisture = moisture / 100.0

        img = Image.open(img_path).convert("RGB")

        # Crop out the top strip (removes the moisture label paper)
        w, h = img.size
        crop_top = int(self.crop_top_frac * h)
        img = img.crop((0, crop_top, w, h))

        if self.transform:
            img = self.transform(img)

        moisture = torch.tensor([moisture], dtype=torch.float32)
        return img, moisture
