import os
from scipy.io import loadmat
from enum import Enum


DATA_ROOT = "../Dataset"
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
