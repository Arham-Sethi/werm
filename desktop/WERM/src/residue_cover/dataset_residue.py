from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ResidueDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, strict=True):
        self.root = Path(root)
        self.df = pd.read_csv(csv_path)

        self.df["photo_id"] = self.df["photo_id"].astype(str).str.strip()

        exists_mask = self.df["photo_id"].apply(lambda n: (self.root / n).exists())
        missing = self.df.loc[~exists_mask, "photo_id"].tolist()

        self.df = self.df[exists_mask].reset_index(drop=True)

        if strict and len(missing) > 0:
            print(f"[ResidueDataset] Dropped {len(missing)} rows with missing files. Example:", missing[:5])

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["photo_id"]

        image = Image.open(img_path).convert("RGB")
        y = float(row["fR"])  # already 0–1 fraction

        if self.transform:
            image = self.transform(image)

        return image, y, row["photo_id"]
