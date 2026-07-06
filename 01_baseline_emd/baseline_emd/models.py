import numpy as np
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class MatlabSVMQ(SVC):
    def fit(self, X, y):
        variance = np.var(X, axis=0).mean() + 1e-12
        gamma_matlab = 1.0 / (X.shape[1] * variance)
        self.gamma = gamma_matlab
        self.degree = 2
        self.coef0 = 1
        self.C = 1.0
        return super().fit(X, y)


def build_models():
    return {
        "SVM-Q": MatlabSVMQ(kernel="poly"),
        "KNN-W": KNeighborsClassifier(weights="distance"),
        "LDA": LinearDiscriminantAnalysis(),
        "RF": RandomForestClassifier(random_state=0),
        "LGBM": LGBMClassifier(random_state=0),
    }
