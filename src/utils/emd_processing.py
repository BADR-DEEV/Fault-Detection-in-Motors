from PyEMD import EMD
import numpy as np

def sig_to_imf(sig, max_imfs=10):
    emd = EMD()

    # Run EMD decomposition
    emd.emd(sig)

    # Extract IMFs + residue
    imfs, residue = emd.get_imfs_and_residue()

    # If no IMFs exist, return empty
    if imfs is None or imfs.size == 0:
        return np.array([]), np.array([]), sig

    # Ensure 2D
    imfs = np.atleast_2d(imfs)

    # Limit to max_imfs (IMF1 ... IMF10)
    imfs = imfs[:max_imfs]

    # ❗ DROP IMF1 (noisiest)
    if imfs.shape[0] > 1:
        imfs_filtered = imfs[1:]      # IMF2 → IMF10
    else:
        imfs_filtered = np.array([])

    # Reconstruct cleaned signal
    if imfs_filtered.size > 0:
        reconstructed = np.sum(imfs_filtered, axis=0) + residue
    else:
        reconstructed = residue.copy()

    return imfs_filtered, residue, reconstructed
