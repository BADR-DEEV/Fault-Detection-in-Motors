from scipy.io import loadmat
from enum import Enum

HealthyDirectory = "Dataset/Healthy"
FaultyDirectory = "Dataset/Faulty"
# directoryName = Enum("directoryName", "Healthy Faulty")

def load_file(file_path :str, directory : str):
    data = loadmat(f'Dataset/{directory}/{file_path}')
    return data
