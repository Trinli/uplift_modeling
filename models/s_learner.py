"""
Implementation of the model proposed by Lo (2002).
Later called the "S-learner", although the s-learner
does not define what base learner is used, and Lo
used logistic regression. 
Actually, the S-learner might just include t as any
other variable, whereas the stuff proposed by Lo
explicitly includes the interaction t*X.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class SLearner():
    """
    """
    def __init__(self, base_learner):
        """
        Note: Linear regression or Logistic Regression will
        not produce a functioning uplift model. E.g. linear
        regression will predict the identical value for every
        observation, logistic regression would predict an
        uplift that is dependent on the conversion rate without
        treatment.
        Use some non-linear model, like a neural net.
        """
        self.model = base_learner()

    def fit(self, X, y, t):
        tmp_x = np.concatenate([X, t], axis=1)
        self.model.fit(tmp_x, y)

    def predict_uplift(self, X):
        tmp_X_t = np.concatenate([X, np.ones(shape=X.shape[0])], axis=1)
        tmp_X_c = np.concatenate([X, np.zeros(shape=X.shape[0])], axis=1)
        pred_t = self.model.predict(tmp_X_t)
        pred_c = self.model.predict(tmp_X_c)
        return pred_t - pred_c


class LoUplift():
    """
    Uplift model following Lo (2002). It is an interaction model
    with features X and X*t with a logistic regression model as
    base learner.
    """
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.model_pos_idx = None

    def fit(self, X, y, t):
        """
        Args:
        X (np.array): Features
        y (np.array): Class-labels
        t (np.array): Treatment-labels
        """
        tmp_X = np.concatenate([X, X * t[:, np.newaxis], t[:, np.newaxis]], axis=1)
        self.model.fit(tmp_X, y)
        if self.model.classes_[0] == True:
            self.model_pos_idx = 0
        else:
            self.model_pos_idx = 1

    def predict_uplift(self, X):
        tmp_X_t = np.concatenate([X, X, np.ones(shape=(X.shape[0], 1))], axis=1)
        tmp_X_c = np.concatenate([X, np.zeros(shape=X.shape), np.zeros(shape=(X.shape[0], 1))], axis=1)
        prob_t = self.model.predict_proba(tmp_X_t)[:, self.model_pos_idx]  # Oops. This returns one probability for positive, one for negative.
        prob_c = self.model.predict_proba(tmp_X_c)[:, self.model_pos_idx]
        return prob_t - prob_c
