# ood.py

import numpy as np


class MahalanobisOOD:
    """
    Industry-standard novelty detection
    """
    def __init__(self):
        self.mean = None
        self.cov_inv = None

    def fit(self, embeddings):
        self.mean = embeddings.mean(axis=0)
        cov = np.cov(embeddings, rowvar=False)
        self.cov_inv = np.linalg.pinv(cov)

    def distance(self, x):
        d = x - self.mean
        return float(d @ self.cov_inv @ d.T)

    def probability(self, dist, threshold):
        return np.exp(-dist / threshold)
