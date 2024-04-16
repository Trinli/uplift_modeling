"""
Double classifier for uplift using Gaussian Process classifier as base learner.
"""

import sklearn.gaussian_process as GP  # Need to make an uplift model with these.
from sklearn.gaussian_process.kernels import RBF

class GaussianProcessUplift():
    """
    """
    def __init__(self):
        """
        Initialize t- and c-models.
        """
        self.model_t = GP.GaussianProcessClassifier(kernel=RBF(1.0))
        self.model_c = GP.GaussianProcessClassifier(kernel=RBF(1.0))
    
    def fit(self, X_t, y_t, X_c, y_c):
        """
        Args:
        X_t (np.array): Features of treated observations
        y_t (np.array): Labels of treated observations
        X_c (np.array): Features of untreated observations
        y_c (np.array): Labels of untreated observations
        """
        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)
    
    def predict_uplift(self, X):
        """
        """
        tmp_t = self.model_t.predict_proba(X)
        # Figure out which is the positive class:
        if self.model_t.classes_[0]:
            # One of the two items in classes_ is True
            t_idx = 0
        else:
            t_idx = 1
        tmp_c = self.model_c.predict_proba(X)
        if self.model_c.classes_[0]:
            c_idx = 0
        else:
            c_idx = 1

        uplift = tmp_t[:, t_idx] - tmp_c[:, c_idx]  # Does this work?
        return uplift

    def generate_sample():
        pass

    def predict_uncertainty():
        pass

    def get_credible_intervals():
        pass
