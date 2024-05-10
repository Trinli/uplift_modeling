"""
This is a simple benchmark model that is something that could be
done without any new theory for uplift modeling.

The model is a double classifier. First, the dataset is split
into treated and untreated observations. Then the treatment-pipeline
is trained. The pipeline consists of undersampling, then a model
(in this case logistic regression), and lastly calibration which
is here performed with isotonic regression.
Suitable undersampling parameters are found using cross-validation
and a metric suitable for classification (AUC-ROC in this case).
The best model is kept. The same process is repeated for the
un-treated observations.
These two pipelines are then combined to make uplift predictions.
"""
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

class CalibratedClassifier():
    """
    A classifier handling search for undersampling
    parameter and cross-validation. Results in a classifier
    that produces well-calibrated predictions.
    """
    def __init__(self):
        self.model = None
        self.calibration = None
        self.k = 1
        self.pos_idx = None  # Index for positive predictions from model

    def fit(self, X, y, X_val, y_val):
        """
        Method for fitting a calibrated classifier. Performs undersampling
        and selects the best model based on auc-roc.

        Args:
        """
        # Values to search for optimal undersampling parameter in
        k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        max_AUC_ROC = 0.0
        for k in k_values:
            # 1. Undersample data
            try:
                tmp_X, tmp_y = classic_undersampling(X, y, k)
            except AssertionError:
                print("Not enough data for k={}".format(k))
                continue
            # 2. Train model
            tmp_model = LogisticRegression()
            tmp_model.fit(tmp_X, tmp_y)
            if tmp_model.classes_[0]:
                tmp_pos_idx = 0
            else:
                tmp_pos_idx = 1
            # 3. Calibrate model
            # Predictions for calibration (same dataset? Sure)
            tmp_pred = tmp_model.predict_proba(X_val)
            tmp_pred = tmp_pred[:, tmp_pos_idx]
            tmp_calibration_model = IsotonicRegression(y_min=0.0000001, y_max=0.999999,
                                                       out_of_bounds='clip')
            tmp_calibration_model.fit(tmp_pred, y_val)
            # 4. Estimate metrics
            tmp_calibrated_pred = tmp_calibration_model.predict(tmp_pred)
            tmp_auc_roc = roc_auc_score(y_val, tmp_calibrated_pred)
            #  -if better than previous, store in self.
            if tmp_auc_roc > max_AUC_ROC:
                self.model = deepcopy(tmp_model)
                self.calibration_model = deepcopy(tmp_calibration_model)
                self.k = k
                self.pos_idx = tmp_pos_idx
                max_AUC_ROC = tmp_auc_roc

    def predict(self, X):
        """
        Method for predicting class probability (for the positive class)

        Args:
        X (np.array): Features of the data
        """
        tmp_pred = self.model.predict_proba(X)
        # Pick positive class from predictions and return a 1-d array.
        tmp_pred = tmp_pred[:, self.pos_idx]
        tmp_calibrated_pred = self.calibration_model.predict(tmp_pred)
        return tmp_calibrated_pred


class NaiveDoubleClassifier():
    """
    Model using two classic calibrated classifiers to do uplift
    modeling. This is basically two separate pipelines for predicting
    class probabilities with calibration and the predictions are
    combined at a final step to predict uplift.

    Fitting uses the validation set to fit a calibration method, and further
    uses the same for model selection between models trained with different
    undersampling factors.
    """
    def __init__(self):
        self.model_t = CalibratedClassifier()
        self.model_c = CalibratedClassifier()
        self.k_t = None
        self.k_c = None

    def fit(self, data):
        """
        Args:
        data (load_data.DatasetCollection): Data in classic format
        """
        tmp_X_t = data['training_set', None, 'treatment']['X']
        tmp_y_t = data['training_set', None, 'treatment']['y']
        tmp_X_val_t = data['validation_set', None, 'treatment']['X']
        tmp_y_val_t = data['validation_set', None, 'treatment']['y']
        self.model_t.fit(tmp_X_t, tmp_y_t, tmp_X_val_t, tmp_y_val_t)
        self.k_t = self.model_t.k
        tmp_X_c = data['training_set', None, 'control']['X']
        tmp_y_c = data['training_set', None, 'control']['y']
        tmp_X_val_c = data['validation_set', None, 'control']['X']
        tmp_y_val_c = data['validation_set', None, 'control']['y']
        self.model_c.fit(tmp_X_c, tmp_y_c, tmp_X_val_c, tmp_y_val_c)
        self.k_c = self.model_c.k
        print("Optimal k-values: [{}, {}]".format(self.k_t, self.k_c))

    def predict_uplift(self, X):
        """
        Args:
        X (np.array): Features to predict from
        """
        return self.model_t.predict(X) - self.model_c.predict(X)


def classic_undersampling(X, y, k, seed=None):
    """
    Function for regular undersampling for classification.
    Negative observations are randomly dropped to satisfy
    p_{new}(y) = k * p_{old}(y).

    Args:
    'X' (np.array): Features of training data
    'y' (np.array): Labels of training data
    k (float): Undersampling factor. Must be larger than zero.
    seed (int): Seed for random number generation if any.
    """
    # Number of positives in group:
    num_pos = sum(y)
    # Find indices for all positive samples
    pos_idx = np.array([i for i, tmp in enumerate(y) if tmp])
    num_neg = sum(~y)
    # Find indices for all negative samples:
    neg_idx = np.array([i for i, tmp in enumerate(y) if not tmp])
    num_tot = len(y)
    assert k * num_pos / num_tot < 1, "Not enough negative samples for selected k"
    if k >= 1:
        # Drop negative samples for k >= 1:
        num_neg_new = max(0, int(num_tot / k) - num_pos)
        num_pos_new = num_pos
    elif 1 > k_t > 0:  # Aiming for k * pos/tot = pos_new / tot_new
        # Drop positive samples for k < 1:
        num_neg_new = num_neg  # stays constant
        num_pos_new = max(0, int(k * num_pos / num_tot * num_neg /\
            (1 - k * num_pos / num_tot)))
    else:
        raise ValueError("k needs to be larger than 0.")

    # Create indices for sampling:
    new_neg_idx = np.random.choice(neg_idx, size=num_neg_new, replace=False)
    new_pos_idx = np.random.choice(pos_idx, size=num_pos_new, replace=False)
    idx = np.concatenate([new_neg_idx, new_pos_idx], axis=0)

    if seed is not None:
        # Set random seed.
        # Using a separate random number generator not to mess
        # with the global RNG of Numpy.
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    else:
        # Do as was done before the change with random number
        # generators in Numpy (1.20.1?).
        np.random.shuffle(idx)
    tmp_X = X[idx, :]
    tmp_y = y[idx]

    return tmp_X, tmp_y
