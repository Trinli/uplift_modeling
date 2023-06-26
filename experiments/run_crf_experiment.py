"""
Experiements using causal random forest by Athey & al.

Adding functions for k-undersampling -related experiments.
"""

import gc
import os
import sys
import time
import rpy2.robjects as robjects
import numpy as np
import models.uplift_calibration as uplift_calibration  # Used for analytic correction.
import data.load_data as load_data
import data.pickle_dataset as pickle_dataset
import metrics.uplift_metrics as uplift_metrics
from sklearn.linear_model import LogisticRegression


def numpy_to_csv(data, filename, path='./tmp/'):
    """
    Function for converting numpy array to csv
    -skip headers

    Args:
    data (?): Data in some format. Should it be a dict
    with X, y, and t or just one of these?

    Returns:
    Nothing. Data is saved to disk.
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

def numpy_X_to_csv(data, filename, path='./tmp/', set='test'):
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

def csv_to_numpy(filename, path='./tmp/', k=1, delimiter=','):
    """
    Used to read results (predictions) from R.
    """
    tmp = np.genfromtxt(path + filename + '.' + str(k)
                        + '.predictions.csv', delimiter=delimiter)
    return tmp


def read_results(filename, path='./tmp/'):
    return csv_to_numpy(filename + 'results.csv', path)


def crf_model(data, filename, k=None):
    """
    Wrapper for model() to train and evaluate a crf-model.
    Handles extraction of training set with or without
    k-undersampling.

    Args:
    data (load_data.DatasetCollection): Training, validation, and
     testing set.
    filename (str): Name and location of data file (.csv)
    k (int): If 'None', data is used as is. Otherwise k-undersampling
     is preformed.

    Returns:
    Dict of testing and validation set results.
    """
    if k is None:
        X = data['training_set']['X']
        y = data['training_set']['y']
        t = data['training_set']['t']
        training_set = {'X': X, 'y': y, 't': t}
    else:
        # k-undersampling:
        training_set = experiment_aux.get_training_set(data, k, 'natural')
    return train_and_evaluate_model(training_set, data, filename)  # THIS DOES NOT CONFORM TO SIGNATURE


def train_and_evaluate_model(training_set, data, filename, k):
    """
    Function for _both_ training and evaluating a crf-model.

    Args:
    training_set (dict): Dict with keys 'X', 'y', and 't'. This is used
     for training the models etc. Used to change training set and still
     validate on original data.
    data (load_data.DatasetCollection): Dataset with training, testing,
     and validation sets.
    filename (str): Name of data being processed. Stored in metrics-object.
    k (float): k used to train the model. Stored in metrics-object.

    Returns:
    [uplift_metrics.UpliftMetrics, int]: List of optimal model testing set metrics and related k.
    """
    # Move over training data:
    # Save training set to csv's
    numpy_to_csv(training_set, filename)

    #robjects.r("source('model_crf_experiment.R')")  # Where is this now?
    robjects.r("source('models/uplift_random_forest.R')")  # Is this correct?
    robjects.r("train_crf('{}')".format(filename))

    # Now store validation or testing set
    # Loop this to not cause R to crash... and maybe
    # don't reload model if it is in memory?
    bins = 100
    pred = np.array([])
    r_command = "model = load_model('{}')".format(filename + str(1))
    robjects.r(r_command)

    # Estimate validation set metrics:
    n_validation_samples = len(data['validation_set']['y'])
    print("Estimating validation set metrics...")
    for i in range(bins):
        # Do this in batches.
        tmp_data = data['validation_set']['X'][int(i * n_validation_samples/bins):int((i + 1) * n_validation_samples/bins)]
        numpy_X_to_csv(tmp_data, filename)
        # Change 'predict_crf' to not load model.
        robjects.r("predict_crf('{}', load_model=FALSE)".format(filename))
        # Add results to some list
        tmp = csv_to_numpy(filename)
        pred = np.concatenate((pred, tmp), axis=0)
        #print("{} out of {} done.".format(i + 1, bins))
    validation_set_metrics = uplift_metrics.UpliftMetrics(
        data['validation_set']['y'],
        pred,
        data['validation_set']['t'],
        test_name='Initial CRF experiment (validation_set)',
        algorithm='causal random forest',
        dataset='{} with sampling rate natural'.format(filename),
        parameters=None)

    # Estimate testing set metrics:
    pred = np.array([])  # Reset array.
    n_testing_samples = len(data['testing_set']['y'])
    print("Estimting testing set metrics")
    for i in range(bins):
        # Do this in batches.
        tmp_data = data['testing_set']['X'][int(i * n_testing_samples/bins):int((i + 1) * n_testing_samples/bins)]
        numpy_X_to_csv(tmp_data, filename)
        # Change 'predict_crf' to not load model.
        robjects.r("predict_crf('{}', load_model=FALSE)".format(filename))
        # Add results to some list
        tmp = csv_to_numpy(filename)
        pred = np.concatenate((pred, tmp), axis=0)
        #print("{} out of {} done.".format(i + 1, bins))
    print("Length of predict-vector")
    print(len(pred))
    print("Number of testing_samples: {}".format(n_testing_samples))
    assert len(pred) == n_testing_samples, "Error in predictions."
    testing_set_metrics = uplift_metrics.UpliftMetrics(
        data['testing_set']['y'],
        pred,
        data['testing_set']['t'],
        test_name='CRF experiment vs. dataset size',
        algorithm='causal random forest',
        dataset='{} with natural sampling rate'.format(filename),
        parameters='k {}'.format(k))
    # Don't store metrics here. Let the k-undersampling search function do that.
    # metrics.write_to_csv()
    print(validation_set_metrics)
    return {'testing_set_metrics': testing_set_metrics,
            'validation_set_metrics': validation_set_metrics}


def k_undersamping_crf(data, k_min=-1, k_max=50, k_step=1,
                       model_id='',
                       metrics_file='datasets/criteo2/generated_subsamples/uplift_results.csv',
                       filename='tmp'):
    print("Running crf-experiment with k-undersampling. Model id: {}".format(model_id))

    results = []  # List to store resulting metrics

    best_improvement_to_random = float("-inf")
    best_k = -1
    kvalues = list(range(k_min, k_max, k_step))
    print("Sweeping over kvalues=%s" % kvalues)

    for i in kvalues:
        gc.collect()
        start = time.time()
        print("-" * 40)
        print("k: {}".format(i))

        try:
            training_set = experiment_aux.get_training_set(data, i, 'natural')
        except AssertionError:
            print("k {} causes # negative samples to drop to zero".format(i))
            break
        # Separate treatment and control samples
        tmp_metrics = train_and_evaluate_model(training_set, data, filename, i)

        metrics = tmp_metrics['validation_set_metrics']
        if metrics.improvement_to_random > best_improvement_to_random:
            best_improvement_to_random = metrics.improvement_to_random
            best_metrics = tmp_metrics
            best_k = i

        print("Evaluated in %.2f seconds" % (time.time()-start))
        print(metrics)
        results.append(tmp_metrics)

    # Estimate testing set metrics:
    testing_set_metrics = best_metrics['testing_set_metrics']
    testing_set_metrics.write_to_csv(file_=metrics_file)

    print("k-CRF testing set metrics")
    print("best k: {}".format(best_k))
    print(best_metrics['testing_set_metrics'])

    return [metrics, best_k]


# Code for experiments with uplift RF as well (Guelman & al. 2015).
def uplift_rf_model(training_set, filename):
    """
    Wrapper for training uplift RF model (Guelman & al 2015).
    Uplift RF is coded in 
    """
    robjects.r("library(uplift)")
    numpy_to_csv(training_set, filename)


def crf_experiment(filename='criteo_100k.csv', fraction_of_data=1.0):

    # Load dataset:
    if filename == 'criteo_100k.csv':
        print("This is just a test run with a small dataset.")
        data = load_data.get_criteo_test_data()
    elif filename.endswith('.csv'):
        # Default parameters?
        if 'voter' in filename:
            print("The dataset is the Voter-dataset with 1:1 and .5 modifications.")
            data = load_data.DatasetCollection(file_name=filename,
                                               data_format=load_data.VOTER_FORMAT)
        elif 'zenodo' in filename:
            print("The dataset is a modified Zenodo-dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.ZENODO_FORMAT)
        elif 'hillstrom' in filename.lower():
            print("The dataset is the Hillstrom dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.HILLSTROM_CONVERSION)
        elif 'starbucks' in filename:
            print("The dataset is the Starbucks dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.STARBUCKS_FORMAT)
        else:
            print("Assuming that the dataset is a criteo-dataset.")
            data = load_data.DatasetCollection(file_name=filename)
    else:
        data = pickle_dataset.load_pickle(filename)

    # Subsample training data:
    n_samples = len(data['training_set']['y'])
    idx = np.random.choice(range(0, n_samples),
                           size=int(n_samples * fraction_of_data),
                           replace=False)
    X = data['training_set']['X'][idx, :]
    y = data['training_set']['y'][idx]
    t = data['training_set']['t'][idx]
    training_set = {'X': X, 'y': y, 't': t}

    # Move over training data:
    # Save training set to csv's
    numpy_to_csv(training_set, filename)

    #robjects.r("source('model_crf_experiment.R')")
    robject.r("source('models/uplift_random_forest.R')")
    robjects.r("train_crf('{}')".format(filename))

    # Now store validation or testing set
    # Loop this to not cause R to crash... and maybe
    # don't reload model if it is in memory?
    bins = 100
    pred = np.array([])
    r_command = "model = load_model('{}')".format(filename + str(1))
    robjects.r(r_command)
    n_testing_samples = len(data['testing_set']['y'])
    for i in range(bins):
        tmp_data = data['testing_set']['X'][int(i * n_testing_samples/bins):int((i + 1) * n_testing_samples/bins)]
        numpy_X_to_csv(tmp_data, filename)
        # Change 'predict_crf' to not load model.
        robjects.r("predict_crf('{}', load_model=FALSE)".format(filename))
        # Add results to some list
        tmp = csv_to_numpy(filename)
        pred = np.concatenate((pred, tmp), axis=0)
        print("{} out of {} done.".format(i + 1, bins))

    print("Lenght of predictions: {}".format(len(pred)))
    print("Length of testing_samples: {}".format(n_testing_samples))
    assert len(pred) == n_testing_samples, "Error in predictions."
    metrics = uplift_metrics.UpliftMetrics(
        data['testing_set']['y'],
        pred,
        data['testing_set']['t'],
        test_name='Initial CRF experiment',
        algorithm='causal random forest',
        dataset='{} with sampling rate natural'.format(filename),
        parameters=None)
    # Store results
    metrics.write_to_csv()
    print(metrics)


def uplift_rf_experiment(filename='criteo_100k.csv',
                         path='./',
                         k_t=1, k_c=1):
    """
    Function for testing uplift random forest.
    -Correction included?
    Try only uplift rf first.

    Args:
    filename (str): Name of data file. Pickle or csv.
    k_t (float): undersampling "factor" for treated samples (1 equals no change)
    k_c (float): undersampling "factor" for untreated (control) samples

    Misc:
    -If filename contains all info on experiment, no additional 'k' or other
     info might be needed(?).
    -Here we are testing the models using the validation set and leave the testing
     set untouched for later.
    """
    print("Running Uplift Random Forest for {}".format(filename))
    # Load dataset:
    if filename == 'criteo_100k.csv':
        print("This is just a test run with a small dataset.")
        data = load_data.get_criteo_test_data()
    elif filename.endswith('.csv'):
        if 'voter' in filename:
            print("The dataset is the Voter-dataset with 1:1 and .5 modifications.")
            data = load_data.DatasetCollection(file_name=path + filename,
                                               data_format=load_data.VOTER_FORMAT)
        elif 'zenodo' in filename:
            print("The dataset is a modified Zenodo-dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.ZENODO_FORMAT)
        elif 'hillstrom' in filename.lower():
            print("The dataset is the Hillstrom dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.HILLSTROM_CONVERSION)
        elif 'starbucks' in filename:
            print("The dataset is the Starbucks dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.STARBUCKS_FORMAT)
        else:
            print("Assuming that the dataset is a criteo-dataset.")
            data = load_data.DatasetCollection(file_name=path + filename)
    else:
        # Seed is not set here, i.e. reproducibility is not guaranteed!
        data = pickle_dataset.load_pickle(filename)

    # Process id used to identify files passed between Python and R
    model_id = str(os.getpid())  # As string...
    # Estimate p_t & p_c from training data before undersampling.
    n_pos_t = np.sum(data['training_set']['y'][data['training_set']['t']])
    n_tot_t = np.sum(data['training_set']['t'])
    p_t = n_pos_t / n_tot_t
    n_pos_c = np.sum(data['training_set']['y'][~data['training_set']['t']])
    n_tot_c = np.sum(~data['training_set']['t'])
    p_c = n_pos_c / n_tot_c
    # Subsample training data:
    if k_t != 1 or k_c != 1:
        # Seed selected by fair die.
        training_set = data.undersampling(k_t, k_c, seed=3)
    else:
        X = data['training_set']['X']
        y = data['training_set']['y']
        t = data['training_set']['t']
        # numpy_to_csv() takes dict as input:
        training_set = {'X': X, 'y': y, 't': t}

    # Move over training data:
    # Save training set to csv's
    numpy_to_csv(training_set, model_id)  # Add some ID here.

    #robjects.r("source('model_crf_experiment.R')")
    robjects.r("source('models/uplift_random_forest.R')")
    robjects.r("model = train_uplift_rf('{}')".format(model_id))

    # Now store validation or testing set
    # Loop this to not cause R to crash... and maybe
    # don't reload model if it is in memory?
    bins = 100
    #pred = np.array([[]])
    pred = np.empty(shape=[0, 2])
    # Model already in memory after training uplift rf.
    #r_command = "model = load_model('{}')".format(filename + str(1))
    #robjects.r(r_command)
    n_testing_samples = len(data['validation_set']['y'])
    for i in range(bins):
        tmp_data = data['validation_set']['X'][int(i * n_testing_samples/bins):int((i + 1) * n_testing_samples/bins)]
        numpy_X_to_csv(tmp_data, model_id)
        # Change 'predict_crf' to not load model.
        robjects.r("predict_uplift_rf('{}', load_model=FALSE)".format(model_id))
        # Everything should be fine uptil now. predict_uplift_rf() stores data
        # as "table". We will need some adjustments to read it in, though.
        # Add results to some list
        tmp = csv_to_numpy(model_id, delimiter=' ')
        pred = np.concatenate((pred, tmp), axis=0)
        print("{} out of {} done.".format(i + 1, bins))

    print("Lenght of predictions: {}".format(len(pred)))
    print("Length of testing_samples: {}".format(n_testing_samples))
    assert len(pred) == n_testing_samples, "Error in predictions."
    # Uplift without any corrections:
    # treated conversion probability is in col 0:
    # WE DON'T HAVE p_t nor k_t here.... FIX.
    # here pred[:, 0] refers to output of the R code, i.e. the t-class.
    prob_t = uplift_calibration.analytic_correction(pred[:, 0], k_t, p_t)  # k_t and p_t not defined.
    # can also use signature analytic_correction(prob[:, 0], s=...)
    prob_c = uplift_calibration.analytic_correction(pred[:, 1], k_c, p_c)
    tau = np.array([item1 - item2 for item1, item2 in zip(prob_t, prob_c)])
    # tau = np.array([item1 - item2 for item1, item2 in pred])

    # Uplift RF returns both conversion probabilities, not uplift.
    # This needs to be dealt with both in "save_results (in R-code)"
    # and here.

    metrics = uplift_metrics.UpliftMetrics(
        data['validation_set']['y'],
        tau,
        data['validation_set']['t'],
        test_name='Uplift RF experiment (grid-search)',
        algorithm='uplift random forest',
        dataset='{} with k_t {} and k_c {}'.format(filename, k_t, k_c),
        parameters=[k_t, k_c])  # Parameters not written to csv.
    # Store results in it's own file (parse later)
    metrics.write_to_csv(file_='./results_uplift_rf/grid_search/results.csv')
    print(metrics)

    # Estimate testing set metrics, write to file, and parse later.

    bins = 100
    pred = np.empty(shape=[0, 2])
    # Model already in memory after training uplift rf.
    #r_command = "model = load_model('{}')".format(filename + str(1))
    #robjects.r(r_command)
    n_testing_samples = len(data['testing_set']['y'])
    for i in range(bins):
        tmp_data = data['testing_set']['X'][int(i * n_testing_samples/bins):int((i + 1) * n_testing_samples/bins)]
        numpy_X_to_csv(tmp_data, model_id)
        # Change 'predict_crf' to not load model.
        robjects.r("predict_uplift_rf('{}', load_model=FALSE)".format(model_id))
        # Everything should be fine uptil now. predict_uplift_rf() stores data
        # as "table". We will need some adjustments to read it in, though.
        # Add results to some list
        tmp = csv_to_numpy(model_id, delimiter=' ')
        pred = np.concatenate((pred, tmp), axis=0)
        print("{} out of {} done.".format(i + 1, bins))
    # Do the analytic corrections:
    prob_t = uplift_calibration.analytic_correction(pred[:, 0], k_t, p_t)
    prob_c = uplift_calibration.analytic_correction(pred[:, 1], k_c, p_c)
    tau = np.array([item1 - item2 for item1, item2 in zip(prob_t, prob_c)])

    # Estimate metrics
    test_metrics = uplift_metrics.UpliftMetrics(
        data['testing_set']['y'],
        tau,
        data['testing_set']['t'],
        test_name='Uplift RF experiment (grid search)',
        algorithm='uplift random forest',
        dataset='{} with k_t {} and k_c {}'.format(filename, k_t, k_c),
        parameters=[k_t, k_c]
    )
    # Write to file
    test_metrics.write_to_csv(file_='./results_uplift_rf/grid_search/results.csv_testing_set.csv')
    print("Testing set metrics:")
    print(test_metrics)

    return metrics


def uplift_dc_experiment(filename='criteo_100k.csv',
                         path='./',
                         k_t=1, k_c=1):
    """
    Function for testing uplift double classifier
    -Correction included.

    Args:
    filename (str): Name of data file. Pickle or csv.
    k_t (float): undersampling "factor" for treated samples (1 equals no change)
    k_c (float): undersampling "factor" for untreated (control) samples

    Misc:
    -Here we are testing the models using the validation set and leave the testing
     set untouched for later.
    """
    # 1. Read in data from named file
    print("Running Uplift Double Classifier for {}".format(filename))
    # Load dataset:
    if filename == 'criteo_100k.csv':
        print("This is just a test run with a small dataset.")
        data = load_data.get_criteo_test_data()
    elif filename.endswith('.csv'):
        if 'voter' in filename:
            print("The dataset is the Voter-dataset with 1:1 and .5 modifications.")
            data = load_data.DatasetCollection(file_name=path + filename,
                                               data_format=load_data.VOTER_FORMAT)
        elif 'zenodo' in filename:
            print("The dataset is a modified Zenodo-dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.ZENODO_FORMAT)
        elif 'hillstrom' in filename.lower():
            print("The dataset is the Hillstrom dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.HILLSTROM_CONVERSION)
        elif 'starbucks' in filename:
            print("The dataset is the Starbucks dataset.")
            data = load_data.DatasetCollection(file_name=path+filename,
                data_format=load_data.STARBUCKS_FORMAT)
        else:
            print("Assuming that the dataset is a criteo-dataset.")
            data = load_data.DatasetCollection(file_name=path + filename)
    else:
        # Seed is not set here, i.e. reproducibility is not guaranteed!
        data = pickle_dataset.load_pickle(filename)

    # 2. Estimate p_t & p_c from training data before undersampling.
    n_pos_t = np.sum(data['training_set']['y'][data['training_set']['t']])
    n_tot_t = np.sum(data['training_set']['t'])
    p_t = n_pos_t / n_tot_t
    n_pos_c = np.sum(data['training_set']['y'][~data['training_set']['t']])
    n_tot_c = np.sum(~data['training_set']['t'])
    p_c = n_pos_c / n_tot_c

    # 3. Subsample training data:
    if k_t != 1 or k_c != 1:
        # Seed selected by fair die.
        training_set = data.undersampling(k_t, k_c, seed=3)
    else:
        X = data['training_set']['X']
        y = data['training_set']['y']
        t = data['training_set']['t']
        # numpy_to_csv() takes dict as input:
        training_set = {'X': X, 'y': y, 't': t}

    # 4. Train models on undersampled data
    model_t = LogisticRegression(tol=1e-6)
    model_t.fit(training_set['X'][training_set['t']],
                training_set['y'][training_set['t']])
    model_c = LogisticRegression(tol=1e-6)
    model_c.fit(training_set['X'][~training_set['t']],
                training_set['y'][~training_set['t']])
    # 5. Predict conversion probabilities for VALIDATION set
    prob_t = model_t.predict_proba(data['validation_set', None, 'all']['X'])
    if model_t.classes_[1] == True:
        tmp_t_idx = 1
    elif model_t.classes_[0] == True:
        tmp_t_idx = 0
    else:
        raise ValueError('Logistic regression not returning true/false')
    prob_t = prob_t[:, tmp_t_idx]
    if model_c.classes_[1] == True:
        tmp_c_idx = 1
    elif model_c.classes_[0] == True:
        tmp_c_idx = 0
    else:
        print("Logistic regression not outputting true/false.")
        raise ValueError('Logistic regression not returning true/false')
    prob_c = model_c.predict_proba(data['validation_set', None, 'all']['X'])
    prob_c = prob_c[:, tmp_c_idx]

    # 5. Corrections in undersampling_aux.analytic_correction()
    true_prob_t = uplift_calibration.analytic_correction(prob_t, k=k_t, p_bar=p_t)
    true_prob_c = uplift_calibration.analytic_correction(prob_c, k=k_c, p_bar=p_c)
    tau = np.array([item1 - item2 for item1, item2 in zip(true_prob_t, true_prob_c)])

    # 6. Estimate metrics
    metrics = uplift_metrics.UpliftMetrics(
        data['validation_set']['y'],
        tau,
        data['validation_set']['t'],
        test_name='Uplift DC experiment (grid-search)',
        algorithm='uplift double classifier',
        dataset='{} with k_t {} and k_c {}'.format(filename, k_t, k_c),
        parameters=[k_t, k_c])  # Parameters not written to csv.
    # Store results in it's own file (parse later)
    metrics.write_to_csv(file_='./results_uplift_dc/grid_search/results.csv')
    # Validation set metrics
    print("Validation set metrics:")
    print(metrics)
    
    # Estimate testing set metrics, write to file, and parse later.
    test_prob_t = model_t.predict_proba(data['testing_set']['X'])
    test_prob_t = test_prob_t[:, tmp_t_idx]
    test_prob_c = model_c.predict_proba(data['testing_set']['X'])
    test_prob_c = test_prob_c[:, tmp_c_idx]
    test_prob_t = uplift_calibration.analytic_correction(test_prob_t, k=k_t, p_bar=p_t)
    test_prob_c = uplift_calibration.analytic_correction(test_prob_c, k=k_c, p_bar=p_c)
    tau = np.array([item1 - item2 for item1, item2 in zip(test_prob_t, test_prob_c)])
    # Estimate metrics
    test_metrics = uplift_metrics.UpliftMetrics(
        data['testing_set']['y'],
        tau,
        data['testing_set']['t'],
        test_name='Uplift DC experiment (grid search)',
        algorithm='uplift double classifier',
        dataset='{} with k_t {} and k_c {}'.format(filename, k_t, k_c),
        parameters=[k_t, k_c]
    )
    test_metrics.write_to_csv(file_='./results_uplift_dc/grid_search/results.csv_testing_set.csv')
    print("Testing set metrics:")
    print(test_metrics)
    return metrics


if __name__ == '__main__':
    # Run the main program
    # Enter file name as command line argument
    # Datasets are assumed to reside in './datasets/'
    print("-Run program as 'python run_crf_experiment.py data_file.pickle ./ 1.0 {crf, uplift_rf, uplift_dc} (k_t, k_c)'")
    print("-Datasets are assumed to reside in ./datasets/")
    filename = sys.argv[1]
    path = sys.argv[2]
    fraction_of_data = float(sys.argv[3])
    if len(sys.argv) > 4:
        base_learner = sys.argv[4]
    else:
        # Default value for crf.
        # 'crf' contains the option to not use complete training data.
        base_learner = 'crf'
    if len(sys.argv) > 5:
        k_t = float(sys.argv[5])
    if len(sys.argv) > 6:
        k_c = float(sys.argv[6])
    if len(sys.argv) > 7:
        test_name = sys.argv[7]
    else:
        # Set generic name for test
        test_name = 'Generic {}-test.'.format(base_learner)
    print("Processing for {} with data fraction {} for training".format(
        filename, fraction_of_data))
    if base_learner == 'crf':
        print("Running crf-experiment...", end='')
        #metrics = crf_experiment(filename, path, fraction_of_data)  # How could this work as the function does not take path?
        metrics = crf_experiment(path + filename, fraction_of_data)
        print('Done.')
    elif base_learner == 'uplift_rf':
        # Uplift rf training run.
        print("Running uplift RF-experiment...", end='')
        metrics = uplift_rf_experiment(filename, path, k_t, k_c)
        print('Done.')
    elif base_learner == 'uplift_dc':
        # Basic double classifier with logistic regression.
        # Good benchmark. 
        metrics = uplift_dc_experiment(filename, path, k_t, k_c)
        print('Done.')
