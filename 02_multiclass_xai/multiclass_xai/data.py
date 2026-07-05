from pathlib import Path

from scipy.io import loadmat

from .paths import DATA_ROOT


def load_data_custom(file_name: str, directory: str, data_root: Path = DATA_ROOT):
    return loadmat(data_root / directory / file_name)


def section_dataset(data_root: Path = DATA_ROOT):
    healthy = {"fileName": [], "x": [], "label": 0, "predicted": []}
    faulty = {"fileName": [], "x": [], "label": 1, "predicted": []}

    healthy_dir = data_root / "Healthy"
    faulty_dir = data_root / "Faulty"

    for item in sorted(healthy_dir.iterdir()):
        if item.is_file():
            healthy["fileName"].append(item.name)
            healthy["x"].append(load_data_custom(item.name, "Healthy", data_root)["H"].squeeze())

    for item in sorted(faulty_dir.iterdir()):
        if item.is_file():
            faulty["fileName"].append(item.name)
            faulty["x"].append(load_data_custom(item.name, "Faulty", data_root)["H"].squeeze())

    return healthy, faulty


def deterministic_file_folds(n_files: int, n_folds: int):
    folds = [[] for _ in range(n_folds)]
    for index in range(n_files):
        folds[index % n_folds].append(index)
    return folds
