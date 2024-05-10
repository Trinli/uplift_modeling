
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class DCLogisticRegression():
    """
    Version of logistic regression that handles uplift-related stuff.
    """
    def __init__(self):
        self.t_model = LogisticRegression()
        self.c_model = LogisticRegression()

    def fit(self, X_c, y_c, X_t, y_t):
        """
        Method for fitting a double-classifier.

        Args:
        X_c (numpy.array): Features for control data
        y_c (numpy.array): Label for control data
        ...
        """
        self.t_model.fit(X=X_t, y=y_t)
        self.c_model.fit(X=X_c, y=y_c)

    def predict_uplift(self, X):
        """
        Method for predicting uplift.

        Args:
        X (np.array): Features for samples.
        """
        t_prediction = self.t_model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_t_idx = np.where(self.t_model.classes_ == 1.0)[0][0]
        t_pred = t_prediction[:, true_t_idx].astype(np.float64)
        c_prediction = self.c_model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_c_idx = np.where(self.c_model.classes_ == 1.0)[0][0]
        c_pred = c_prediction[:, true_c_idx].astype(np.float64)
        return t_pred - c_pred


class ClassVariableTransformation():
    """
    Version of class-variable transformation that is just a plain model.
    """
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, z):
        """
        Fitting the model, assumes class-variable transformation as input.

        Args:
        X (): Features
        z (np.array): Class-variable transformed dependent variable.
        """
        self.model.fit(X=X, y=z)

    def predict_uplift(self, X):
        """
        Function for predicting uplift, i.e. change in y.

        Args:
        X (np.array): Features
        """
        tmp_prediction = self.model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_idx = np.where(self.model.classes_ == 1.0)[0][0]
        return 2.0 * tmp_prediction[:, true_idx].astype(np.float64) - 1.0


class CVTRandomForest(ClassVariableTransformation):
    """
    Class-variable transformation with random forest.
    """
    def __init__(self):
        self.model = RandomForestClassifier()


class DCRandomForest(DCLogisticRegression):
    """
    Double-classifier approach with random forest.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.t_model = RandomForestClassifier()
        self.c_model = RandomForestClassifier()


class SVM_IR(object):
    """
    Auxiliary class for support vector machine and isotonic regression.
    The classifier is a pipeline of two where a support vector machine
    first predicts a score and isotonic regression further transforms
    this into a probability.
    """
    def __init__(self):
        from sklearn.svm import SVC
        from sklearn.isotonic import IsotonicRegression
        self.svm_model = SVC()
        self.ir_model = IsotonicRegression(y_min=0.001, y_max=0.999,
                                           out_of_bounds='clip')

    #def fit(self, training_set, calibration_set):
    def fit(self, X, y):
        # Using default hyperparameters
        #self.svm_model.fit(X=training_set['X'], y=training_set['y'])
        self.svm_model.fit(X=X, y=y)
        #tmp_cal = self.svm_model.decision_function(X=calibration_set['X'])
        tmp_pred = self.svm_model.decision_function(X=X)
        #self.ir_model.fit(tmp_cal, calibration_set['y'])
        self.ir_model.fit(tmp_pred, y)

    def predict(self, X):
        tmp_score = self.svm_model.decision_function(X)
        tmp_prob = self.ir_model.predict(tmp_score)
        return tmp_prob


class DCSVM():
    """
    Double classifier using SVM and isotonic regression
    """
    def __init__(self):
        self.t_model = SVM_IR()
        self.c_model = SVM_IR()

    # def fit(self, training_dict, calibration_dict):  # Let's just use training set for both training and calibration.
    def fit(self, X_c, y_c, X_t, y_t):
        """
        Method for fitting the SVM-IR combination for double classifier approach.
        Note that training requires both training and calibration set. The
        validation set can be used as calibration set.

        Args:
        training_dict (dict): Dictionary with X_c, y_c, X_t, and y_t for training
         set.
        calibration_dict (dict): Dictionary with X_c, y_c, X_t, and y_t for training
         set. Usually the validation set is inserted here.
        """
        """
        self.t_model.fit({'X': training_dict['X_t'],
        'y': training_dict['y_t']},
                         {'X': calibration_dict['X_t'],
                          'y': calibration_dict['y_t']})
        self.c_model.fit({'X': training_dict['X_c'],
                          'y': training_dict['y_c']},
                         {'X': calibration_dict['X_c'],
                          'y': calibration_dict['y_c']})
        """
        self.c_model.fit(X=X_c, y=y_c)
        self.t_model.fit(X=X_t, y=y_t)


    def predict_uplift(self, X):
        t_pred = self.t_model.predict(X)
        c_pred = self.c_model.predict(X)
        uplift = t_pred - c_pred
        return uplift
