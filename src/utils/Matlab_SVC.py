
import numpy as np
from sklearn.svm import SVC

class MatlabSVMQ(SVC):
    """
    MATLAB-accurate Quadratic SVM:
    - Quadratic polynomial kernel
    - BoxConstraint = C = 1
    - KernelScale = auto → gamma = 1/(p * var(X))
    """
    def fit(self, X, y):
        # Compute MATLAB KernelScale (gamma)
        # MATLAB: gamma = 1 / ( num_features * var(X) )
        v = np.var(X, axis=0).mean() + 1e-12      # mean variance across dimensions
        p = X.shape[1]                            # num features
        gamma_matlab = 1.0 / (p * v)

        # Set kernel parameters EXACTLY like MATLAB
        self.gamma = gamma_matlab
        self.degree = 2          # quadratic
        self.coef0 = 1           # MATLAB uses (coef0 = 1)
        self.C = 1.0

        return super().fit(X, y)