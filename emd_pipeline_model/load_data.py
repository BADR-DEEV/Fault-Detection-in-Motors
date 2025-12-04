import os
import numpy as np
from scipy.io import loadmat
from enum import Enum

import csv



DATA_CLASSES = {
    "normal": 0,
    "horizontal-misalignment": 1,
    "vertical-misalignment": 2,
    "overhang": 3,
    "underhang": 4,
    "imbalance": 5,
}

import os
import numpy as np

DATA_CLASSES = {
    "normal": 0,
    "horizontal-misalignment": 1,
    "vertical-misalignment": 2,
    "overhang": 3,
    "underhang": 4,
    "imbalance": 5,
}

def load_mafulda_dataset(root_dir):
    X = []
    y = []

    for class_name, label in DATA_CLASSES.items():
        folder = os.path.join(root_dir, class_name)

        for file in os.listdir(folder):
            if file.endswith(".csv"):
                path = os.path.join(folder, file)

                features = np.loadtxt(path, delimiter=",")
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)



DATA_ROOT = "../Dataset/"
healthy_dir = "../Dataset/Healthy/"
faulty_dir = "../Dataset/Faulty/"
os.path.join(DATA_ROOT, "Healthy")



healthy_Dict = {"fileName": [], "x": [], "label": 0, "predicted": []}
faulty_Dict = {"fileName": [], "x": [], "label": 1, "predicted": []}

def load_data_custom(file_path: str, directory: str):
    data = loadmat(f"../Dataset/{directory}/{file_path}")
    return data


def section_dataset():
    for i in os.listdir(healthy_dir):
        healthy_Dict["fileName"].append(i)
        healthy_Dict["x"].append(load_data_custom(i, "Healthy")["H"].squeeze())
    for i in os.listdir(faulty_dir):
        faulty_Dict["fileName"].append(i)
        faulty_Dict["x"].append(load_data_custom(i, "Faulty")["H"].squeeze())
    return healthy_Dict, faulty_Dict
