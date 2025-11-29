from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftfreq
import os
from scipy.stats import skew, kurtosis, entropy
import glob
from load_data import section_dataset

def sig_to_imf(sig):
    emd = EMD()
    imfs = emd.emd(sig)

    # No IMFs detected
    if imfs is None or imfs.size == 0:
        return []

    imfs = np.atleast_2d(imfs).squeeze()

    # If only one IMF exists, make sure shape is correct
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]

    # Return only the first 7 IMFs (or fewer if not available)
    max_imfs = 7
    num_imfs = min(imfs.shape[0], max_imfs)

    return imfs[:num_imfs]
