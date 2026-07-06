# datasets/mafaulda_dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from training.preprocessing import preprocess


FAULT_FAMILY_MAP = {
    "normal": 0,
    "imbalance": 1,
    "horizontal-misalignment": 2,
    "vertical-misalignment": 2,
    "misalignment": 2,
    "underhang": 3,
    "overhang": 3,
    "ball_fault": 3,
    "outer_race": 3,
    "inner_race": 3,
    "cage_fault": 3,
}





class MaFauldaNPZDataset(Dataset):
    """
    Loads processed MaFaulDa data (.npz)
    Compatible with process_data_preserving_structure()
    """

    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.files = sorted(self.root.rglob("*.npz"))

        if len(self.files) == 0:
            raise RuntimeError(f"❌ No NPZ files found in {self.root}")

        print(f"✅ Found {len(self.files)} NPZ samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz = np.load(self.files[idx])

        sig = npz["sig"]        # shape: [N, 3]
        label = npz["label"]    # scalar
        rpm = npz["rpm"]        # scalar

        return (
            torch.tensor(sig, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(rpm, dtype=torch.float32),
        )

        


class RawMAFauldaDataset(Dataset):
    """
    MaFaulDa raw CSV loader
    Folder-based labeling (industrial correct)
    """

    def __init__(self, root_dir, fs=50000, verbose=True):
        script_dir = Path(__file__).resolve().parent
    # Assuming this script is in src/inference, go up to root
        ROOT_DIR = script_dir.parent.parent 
        ROOT_DIR= ROOT_DIR / "src/4k_mafulda_structured/"
       
        self.fs = fs
        self.verbose = verbose
        self.root = ROOT_DIR

        self.samples = []

        csv_files = list(self.root.rglob("*.csv"))
        print(f"✅ Found {len(csv_files)} CSV files in {self.root}")
        if len(csv_files) == 0:
            raise RuntimeError("❌ No CSV files found in MaFaulDa directory")

        for csv_path in csv_files:
            label = self._infer_label_from_path(csv_path)

            if label is not None:
                self.samples.append((csv_path, label))
                if verbose:
                    print(f"✔ {csv_path.relative_to(self.root)} → label {label}")

        if len(self.samples) == 0:
            raise RuntimeError("❌ No valid MAFaulda samples found (check folder names)")

        print(f"\n✅ Loaded {len(self.samples)} MaFaulDa samples")

    def _infer_label_from_path(self, path: Path):
        """
        Infer label from folder structure, not filename
        """
        parts = [p.lower() for p in path.parts]

        # Normal has priority
        if "normal" in parts:
            return 0

        # Imbalance
        if "imbalance" in parts:
            return 1

        # Misalignment
        if "horizontal-misalignment" in parts or "vertical-misalignment" in parts:
            return 2

        # Bearing faults (underhang / overhang)
        if "underhang" in parts or "overhang" in parts:
            return 3

        # Bearing subfolders
        for key in ["ball_fault", "outer_race", "inner_race", "cage_fault"]:
            if key in parts:
                return 3

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        df = pd.read_csv(path, header=None)

        # --- Axis detection ---
        if df.shape[1] >= 4:
            # time | X | Y | Z
            sig = df.iloc[:, 1:4].values
        else:
            # single-axis fallback
            sig = df.iloc[:, 1].values[:, None]

        features, rpm = preprocess(sig, self.fs)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(rpm, dtype=torch.float32),
        )
