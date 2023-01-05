"""
All experiments for extension to undersampling paper.

Notes:
Might need to adjust Batch size, learning rate, and stopping
criteria for neural net.
"""
import time

start_time = time.time()
def print_verbose(txt, verbose=False):
    global start_time
    if verbose:
        print("Time for last phase: {} s".format(time.time() - start_time))
        print(txt)
    start_time = time.time()

print_verbose("Importing libraries.")

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import data.load_data as load_data
print_verbose("Importing neural net library")
import models.uplift_neural_net as uplift_neural_net
print_verbose("Importing undersampling experiment_aux")
import models.uplift_calibration as uplift_calibration
print_verbose("Trying to import R")
try:
    import rpy2.robjects as robjects
except:
    print("R not available in this environment.")
    print("Uplift forests cannot be used.")
print_verbose("Imports done.")


class DoubleClassifier():
    """
    Basic class for double classifier with logistic regression.
    """
    def __init__(self):
        """
        Initialize model
        """
        self.model_t = LogisticRegression()
        self.model_c = LogisticRegression()
        self.analytic_correction = False

    def fit(self, X_t, y_t, X_c, y_c,
            k_t=None, k_c=None, p_t=None, p_c=None):
        """
        Fit method
        If k_t, k_c, p_t, p_c are available, the model will make
         an analytic correction in the prediction phase for the
         bias caused by undersampling with these factors.

        Args:
        k_t (float): k_t used in split undersampling. Needed here
         for analytic correction
        k_c (float): k_c used is split undersampling. Needed
         for analytic correction.
        p_t (float): Average conversion rate in treated observations.
        p_c (float): Average conversion rate in untreated observations.

        """
        if k_t is not None and k_c is not None:
            self.analytic_correction = True
            self.k_t = k_t
            self.k_c = k_c
            self.p_t = p_t
            self.p_c = p_c

        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)
        # LogisticRegression predicts values for every class.
        # Store the index for the positive classes.
        if self.model_t.classes_[0] == True:
            self.model_t_pos_idx = 0
        else:
            self.model_t_pos_idx = 1
        if self.model_c.classes_[0] == True:
            self.model_c_pos_idx = 0
        else:
            self.model_c_pos_idx = 1

    def predict_uplift(self, X):
        """
        Basic predict function. Handles sklearn.

        Args:
        X (numpy.array): Features to predict from.
        """
        pred_t = self.model_t.predict_proba(X)
        # Keep only the predictions for the positive class
        pred_t = pred_t[:, self.model_t_pos_idx]
        pred_c = self.model_c.predict_proba(X)
        # Keep positive predictions
        pred_c = pred_c[:, self.model_c_pos_idx]
        if self.analytic_correction:
            pred_t = uplift_calibration.analytic_correction(
                pred_t, k=self.k_t, p_bar=self.p_t
            )
            pred_c = uplift_calibration.analytic_correction(
                pred_c, k=self.k_c, p_bar=self.p_t
            )
        tau = pred_t - pred_c
        return tau


class ClassVariableTransformation():
    """
    Basic class for class-variable transformation.
    """
    def __init__(self):
        """
        """
        self.model = LogisticRegression()

    def fit(self, X, z):
        """
        Args:
        X (numpy.array): Features of the data
        z (numpy.array): The reverted label.

        Notes:
        The class-variable transformation is solely dependent on
        the y and the t variables leaving it unaffected by undersampling.
        Hence we can use the z-variable provided by the 
        load_data.DatasetCollection-class.
        """
        self.model.fit(X, z)
        # Index for positive class
        if self.model.classes_[0] is True:
            self.pos_idx = 0
        else:
            self.pos_idx = 1

    def predict_uplift(self, X):
        """
        This function does the "normalization" of the predictions.
        While this calibrates the predictions, this does not
        affect the rank between samples and hence does not affect
        AUUC.

        Args:
        X (np.array): Features to predict from.
        """
        predictions = self.model.predict_proba(X)
        predictions = predictions[:, self.pos_idx]
        predictions = 2 * predictions - 1
        return predictions


class UpliftNeuralNet(uplift_neural_net.DirectGradientUpliftNN):
    """
    Class to package data conversions etc. to standardize signature (?).
    Although the neural net uses a somewhat more exotic format for the data.

    Need a version that takes in just training and validation sets, maybe as
    dict with appropriate keys? Or just plain X and y separately.
    The undersampling functions return dicts with keys X, y, z, r, and t.
    We need a wrapper for that that can then be fed into the dataloader class.
    Howerver, r needs to be re-evaluated after undersampling.
    """
    def __init__(self, n_features=12):
        """
        Args:
        n_features (int): Number of features
        """
        print("Maybe set batch size, number of epochs etc in uplift_neural_net.py")
        print_verbose("Initializing neural net.")
        super().__init__(n_features=n_features)
        print_verbose("Done initializing super")

    def fit(self, training_set, validation_set):
        """
        The neural net needs the input data as load_data.DatasetWrapper
        objects.

        Args:
        training_set (dict): Dict with keys 'X', 'y', and 'r'
        validation_set (dict): Dict with keys 'X', 'y', and 'r'
        """
        # DatasetWrapper actually takes a dict with keys X, y, and then possibly
        # t, z, and r. So undersampling can be done, then 
        print_verbose("Wrapping datasets")
        training_set = load_data.DatasetWrapper(training_set)
        validation_set = load_data.DatasetWrapper(validation_set)
        # The super() uses the r-variable for training automatically.
        print_verbose("Fitting model")
        super().fit(training_set, validation_set)
        print_verbose("Fitting done")

    def predict_uplift(self, dataset):
        """
        Method for predicting uplift given dict or DatasetWrapper.
        Using "overloading" (checking type and adjusting input as needed) to
        make this method compatible also with the function calls in teh fit-loop.

        Args:
        dataset (np.array, dict or load_data.DatasetWrapper): One set from
         DatasetCollection (e.g. testing set). Needs to contain
         key 'X'.
        """
        if isinstance(dataset, np.ndarray):
            dataset = {'X': dataset}
        if isinstance(dataset, dict):
            dataset = load_data.DatasetWrapper(dataset)
        return super().predict_uplift(dataset)


class UpliftRandomForest():
    """
    Wrapper class for uplift random forest.
    Include analytic correction with a flag (?).
    """
    def __init__(self):
        # Generating name for instance to avoid conflicts  between
        # multiple processes (e.g. tmp-files).
        self.model_id = str(os.getpid())
        self.analytic_correction = False
        self.k_t = None
        self.k_c = None
        self.p_t = None
        self.p_c = None

    # Add some auxiliary functions to communicate with R
    def numpy_to_csv_(self, data, filename, path='./tmp/'):
        """
        Function for converting numpy array to csv
        -skip headers

        Args:
        data (?): Data in some format. Should it be a dict
        with X, y, and t or just one of these?
        """
        # Check if tmp exists:
        if not os.path.isdir(path):
            if os.path.isfile(path):
                # There is a file with this name. Crash.
                raise Exception  # Just generic exception
            else:
                # Create needed directory
                os.mkdir(path)    
        np.savetxt(path + filename+'.train_X.csv', data['X'], delimiter=',') 
        np.savetxt(path + filename+'.train_y.csv', data['y'], delimiter=',') 
        np.savetxt(path + filename+'.train_t.csv', data['t'], delimiter=',') 
        return

    def numpy_X_to_csv_(self, data, filename, path='./tmp/'):
        """
        Function for storing testing or validation set data (X) for
        further processing in R.
        Args:
        data (np.array): duh.
        filename (str): name of _original_ datafile used in experiments.
        used in naming to keep track of which experiment this belongs to.
        path (str):
        set ({'test', 'val'}): One of two values depending on whether we are
        dealing with validation or testing set.
        """
        np.savetxt(path + filename + '.test.csv', data, delimiter=',')
        return

    def csv_to_numpy_(self, filename, path='./tmp/', k=1, delimiter=','):
        """
        Used to read results (predictions) from R.
        """
        tmp = np.genfromtxt(path + filename + '.' + str(k)  # Use self.model_id here instead.
                            + '.predictions.csv', delimiter=delimiter)
        return tmp

    def fit(self, training_set, k_t=None, k_c=None,
            p_t=None, p_c=None):
        """
        Model fitting with signature conforming to other models
        in this file.

        Args:
        training_set (?): Containing 'X', 'y', and 't' for training set.
        """
        if p_t is not None and p_c is not None:
            self.analytic_correction = True
            self.k_t = k_t
            self.k_c = k_c
            self.p_t = p_t
            self.p_c = p_c
        # Where is the numpy_to_csv-function located? COPY or IMPORT!!
        self.numpy_to_csv_(training_set, self.model_id)  # Add some ID here.
        # Load code for uplift random forest in R
        robjects.r("source('./models/uplift_random_forest.R')")
        # Train model. Will automatically look for the file created by
        # numpy_to_csv() with the given model id.
        robjects.r("model = train_uplift_rf('{}')".format(self.model_id))

    def predict_uplift(self, data):
        # Should k_t and k_c be set earlier?
        # We also need p_t and p_c for the correction!!!
        """
        Method for predicting uplift. Corrects for any biases
        introduced by undersampling assuming that k_t and k_c are
        set. k_t=1 and k_c=1 is equivalent to no correction.

        Args:
        data (np.array): Features of data. 
        k_t (float): k_t used in split-undersampling. Used here to correct
         for the bias introduced by undersampling.
        k_c (float): k_c used in split-undersampling. Used here to correct
         for the bias introduced by undersampling.
        """
        # Now store validation or testing set
        # Loop this to not cause R to crash.
        bins = 100
        pred = np.empty(shape=[0, 2])
        # Model already in memory after training uplift rf.
        #r_command = "model = load_model('{}')".format(filename + str(1))
        #robjects.r(r_command)
        n_testing_samples = data.shape[0]
        for i in range(bins):
            tmp_data = data[int(i * n_testing_samples/bins):int((i + 1) * n_testing_samples/bins), :]
            self.numpy_X_to_csv_(tmp_data, self.model_id)
            # Model already loaded.
            robjects.r("predict_uplift_rf('{}', load_model=FALSE)".format(self.model_id))
            # Add results to list
            tmp = self.csv_to_numpy_(self.model_id, delimiter=' ')
            pred = np.concatenate((pred, tmp), axis=0)
            print("{} out of {} done.".format(i + 1, bins))

        assert len(pred) == n_testing_samples, "Error in predictions."
        # Uplift without any corrections:
        # (treated conversion probability is in col 0)
        if self.k_t is not None and self.k_c is not None:  # etc
            prob_t = uplift_calibration.analytic_correction(pred[:, 0], self.k_t, self.p_t)
            prob_c = uplift_calibration.analytic_correction(pred[:, 1], self.k_c, self.p_c)
        else:
            prob_t = pred[:, 0]  # undersampling_experiment_aux.analytic_correction(pred[:, 0])
            prob_c = pred[:, 1]  # undersampling_experiment_aux.analytic_correction(pred[:, 1])
        tau = np.array([item1 - item2 for item1, item2 in zip(prob_t, prob_c)])
        return tau
