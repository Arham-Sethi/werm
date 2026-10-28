from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # Google Drive project root (WERM)
    project_root: Path = Path("/content/drive/MyDrive/WERM")

    # Data (expects Dry/Moderate/Wet under this folder)
    data_root: Path = project_root / "data" / "soil_moisture"

    # Outputs (runs will be saved here)
    out_root: Path = project_root / "outputs" / "runs"
    run_name: str = "moisture_resnet18_v1"

    # Training
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42

    # Split
    train_frac: float = 0.7  # remaining is validation
    stratified: bool = True  # stratified by class

    # Biological proxy labels (emergence-permissive moisture states)
    # Dry -> M0, Moderate -> M1, Wet -> M2
    label_map = {"Dry": 0, "Moderate": 1, "Wet": 2}
    label_names = {
        0: "M0 – Emergence-inhibiting",
        1: "M1 – Marginally permissive",
        2: "M2 – Permissive",
    }
