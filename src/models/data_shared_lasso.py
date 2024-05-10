"""
Variants of the data-shared lasso.
Original DSL proposed by Gross & Tibshirani (2016)

Note that the data-shared lasso is not necessarily suited
for uplift modeling as it is a form of linear regression
and gives no guarantees that the predicted changes in
probabilities lie in [-1, 1].
"""
from numpy import float64, where, sqrt
import numpy as np


class DataSharedLasso(object):
    """
    Original Data-shared lasso as proposed by Gross &
    Tibshirani (2016).
    Training one of these might require cross-validation
     for parameter selection (at least for alpha).
    """
    def __init__(self, dsl_penalty=1/sqrt(2), alpha=.001):
        """
        Args:
        dsl_penalty (float): We are using a uniform dsl-penalty
         as proposed in the original paper.
        alpha (float): The penalty to use for the lasso-regression.
         Setting this too large will cause all coefficients to
         approach zero.
        """
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=alpha)
        self.dsl_penalty = dsl_penalty

    def fit(self, data):
        """
        Function for fitting a Data-shared lasso using the augmented
         approach proposed by Gross & Tibshirani (2016).

        Args:
        data (load_data.DatasetCollection): Data in format
         proposed in load_data.
        """
        # Info on control (c) samples:
        c_rows, c_cols = data['training_set', None, 'control']['X'].shape
        # Info on training (t) samples:
        t_rows, t_cols = data['training_set', None, 'treatment']['X'].shape
        # The augmented approach is basically one where these two form a
        # super-matrix where one takes up the left-upper cells, and the ohter
        # the right-lower cells, filling the reamining cells with 0's.
        tmp_X = np.zeros([t_rows + c_rows, t_cols * 3])
        # X_1 in the paper is now X_c:
        tmp_X[:c_rows, :c_cols] = data['training_set', None, 'control']['X']
        tmp_X[:c_rows, c_cols:(2 * c_cols)] = self.dsl_penalty *\
                                              data['training_set', None, 'control']['X']
        tmp_X[c_rows:, :t_cols] = data['training_set', None, 'treatment']['X']
        tmp_X[c_rows:, (2 * t_cols):] = self.dsl_penalty *\
                                        data['training_set', None, 'treatment']['X']
        tmp_y = np.concatenate([data['training_set', None, 'control']['y'],
                                data['training_set', None, 'treatment']['y']])
        # tmp_y = tmp_y.astype(np.float64)
        self.model.fit(X=tmp_X, y=tmp_y)

    def predict_uplift(self, X):
        """
        Function for predicting uplift

        Args:
        X (numpy.array([float])): An array of data to use for prediction.
         load_data.DatasetCollection contains these in e.g. testing set etc.
        """
        # Here we need to predict conversion probability both with and
        # without treatment:
        n_rows, n_cols = X.shape
        control_X = np.zeros([n_rows, 3 * n_cols])
        control_X[:, :n_cols] = X
        control_X[:, n_cols:(2 * n_cols)] = self.dsl_penalty * X
        treatment_X = np.zeros([n_rows, 3 * n_cols])
        treatment_X[:, :n_cols] = X
        treatment_X[:, (2 * n_cols):] = self.dsl_penalty * X

        control_prob = self.model.predict(control_X)
        treatment_prob = self.model.predict(treatment_X)
        uplift_prob = treatment_prob - control_prob
        return uplift_prob
