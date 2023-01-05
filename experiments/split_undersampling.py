"""
Script to test hypotheses concerning changes in sampling rates
"""
import copy
import gc
import gzip
import pickle
import sys
import time
import numpy as np
import pandas as pd

import metrics.uplift_metrics as uplift_metrics
from models.misc_models import *
from data.data_aux import *


def cvt_experiment(data, k_min=-1, k_max=500, k_step=10, 
                   cvt_class=ClassVariableTransformation, model_id="cvt",
                   test_name=None,
                   file_name=None):
    """
    Function for finding optimal 'k' for undersamling with class-variable
    transformation (or whatever uplift classifier you chose to pass) and
    testing set metrics for said k.
    Args:
    data (load_data.DatasetCollection): Data set
    k_min (int): smallest k to test
    k_max (int): largest k to test
    k_step (int): interval width of k's to test, e.g. if k_min is 1 and
     k_step is five, testable k will be [1, 6, 11, ...].
    cvt_class: Class of uplift classifier
    model_id (str): Description of model tested.
    test_name (str): Name for test to be saved in csv with results..
    file_name (str): Name of dataset file
    """

    print("Runinng cvt_experiment with cvt_class=%s model_id=%s" % (cvt_class, model_id))

    results = []
    testing_set = data['testing_set']
    validation_set = data['validation_set']

    best_improvement_to_random = float("-inf")
    best_k = -1
    kvalues = list(range(k_min, k_max, k_step))
    print("Sweeping over kvalues=%s" % kvalues)

    for i in kvalues:
        gc.collect()
        start = time.time()
        print("-" * 40)
        print("k: {}".format(i))

        model = cvt_class()
        training_set = get_training_set(data, i, '11')
        model.fit(X=training_set['X'], z=training_set['z'])
        predictions = model.predict_uplift(X=validation_set['X'])
        metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                               predictions,
                                               validation_set['t'],
                                               test_name=test_name,
                                               test_description="validation set",
                                               algorithm=model_id,
                                               dataset=file_name,
                                               parameters='k={}'.format(i))
        metrics.write_to_csv()
        if metrics.improvement_to_random > best_improvement_to_random:
            best_improvement_to_random = metrics.improvement_to_random
            best_model = copy.deepcopy(model)
            best_k = i

        print("Evaluated in %.2f seconds" % (time.time() - start))
        print(metrics)
        results.append([model_id, "validation", i]+metrics2row(metrics))

    # Estimate testing set performance
    testing_set_pred = best_model.predict_uplift(X=testing_set['X'])
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(best_k))
    metrics.write_to_csv()
    print("\n\n")
    print("Testing set metrics for CVT and best k")
    print("Best k: {}".format(best_k))
    print(metrics)
    results.append([model_id, "testing", best_k] + metrics2row(metrics))

    results = pd.DataFrame(results).rename(
        columns=dict(enumerate(["MODEL", "SUBSET", "k"] + COLNAMES)))
    return results, testing_set_pred


def dc_experiment(data, k_min=-1, k_max=50, k_step=1, 
                  dc_class=DCLogisticRegression, model_id="dc",
                  test_name=None,
                  file_name=None):
    """
    Function for finding optimal 'k' with double-classifier
    approach and metrics for said k.
    data (load_data.DatasetCollection): Data set
    data (load_data.DatasetCollection): Data set
    k_min (int): smallest k to test
    k_max (int): largest k to test
    k_step (int): interval width of k's to test, e.g. if k_min is 1 and
     k_step is five, testable k will be [1, 6, 11, ...].
    cvt_class: Class of uplift classifier
    model_id (str): Description of model tested.
    test_name (str): Name for test to be saved in csv with results..
    file_name (str): Name of dataset file
    """

    print("Runinng dc_experiment with dc_class=%s model_id=%s" % (dc_class, model_id))

    results = []
    testing_set = data['testing_set']
    validation_set = data['validation_set']

    best_improvement_to_random = float("-inf")
    best_k = -1
    kvalues = list(range(k_min, k_max, k_step))
    print("Sweeping over kvalues=%s" % kvalues)

    for i in kvalues:
        gc.collect()
        start = time.time()
        print("-" * 40)
        print("k: {}".format(i))

        model = dc_class()
        training_set = get_training_set(data, i, 'natural')
        # Separate treatment and control samples
        X_t = training_set['X'][training_set['t'], :]
        y_t = training_set['y'][training_set['t']]
        X_c = training_set['X'][~training_set['t'], :]
        y_c = training_set['y'][~training_set['t']]
        model.fit(X_c, y_c, X_t, y_t) 

        predictions = model.predict_uplift(validation_set['X'])
        metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                               predictions,
                                               validation_set['t'],
                                               test_name=test_name,
                                               test_description="validation set",
                                               algorithm=model_id,
                                               dataset=file_name,
                                               parameters='k={}'.format(i))
        metrics.write_to_csv()

        if metrics.improvement_to_random > best_improvement_to_random:
            best_improvement_to_random = metrics.improvement_to_random
            best_model = copy.deepcopy(model)
            best_k = i

        print("Evaluated in %.2f seconds" % (time.time() - start))
        print(metrics)
        results.append([model_id, "validation", i] + metrics2row(metrics))

    # Estimate testing set metrics:
    testing_set_pred = best_model.predict_uplift(testing_set['X'])
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(best_k))
    metrics.write_to_csv()

    results.append([model_id, "testing", best_k] + metrics2row(metrics))
    print("DC testing set metrics")
    print("best k: {}".format(best_k))
    print(metrics)

    results = pd.DataFrame(results).rename(
                columns=dict(enumerate(["MODEL", "SUBSET", "k"] + COLNAMES)))
    return results, testing_set_pred


def run_experiment(path, model, k_min,  k_max, k_step):
    """
    Returns a pandas data frame with metrics for each k in range.
    Results will also be written to this directory.
    Args:
    path (str): Path to dataset including file name.
    model (str): Model to be trained.
     'cvt' = class-variable transformation with logistic regression
     'cvtrf' = class-variable transformation with random forest
     'dc' or 'dclr' = double-classifier with logistic regression
     'dcrf' = double-classifier with random forest as base learner
    k_min (int): smallest k to test
    k_max (int): largest k to test
    k_step (int): interval width of k's to test, e.g. if k_min is 1 and
     k_step is five, testable k will be [1, 6, 11, ...].
    """

    fp = gzip.open(path, 'rb') if path.endswith(".gz") else open(path, 'rb')
    data = pickle.load(fp)
    fp.close()

    if model == "cvt":
        results, test_predictions = cvt_experiment(data, k_min, k_max, k_step, ClassVariableTransformation, "cvt",
                                                   test_name="Class-variable transformation with LR and undersampling",
                                                   file_name=path)
    elif model == "cvtrf":
        results, test_predictions = cvt_experiment(data, k_min, k_max, k_step, CVTRandomForest, "cvtrf",
                                                   test_name="Class-variable transformation with RF and undersampling",
                                                   file_name=path)
    elif model == "dc" or model == "dclr":
        results, test_predictions = dc_experiment(data, k_min, k_max, k_step, DCLogisticRegression, "dc",
                                                  test_name="Double-classifier with LR and undersampling",
                                                  file_name=path)
    elif model == "dcrf":
        results, test_predictions = dc_experiment(data, k_min, k_max, k_step, DCRandomForest, "dcrf",
                                                  test_name="Double-classifier with RF and undersampling",
                                                  file_name=path)

    else:
        raise ValueError("Unknown model name=%s! Try: cvt/cvtrf/dclr==dc/dcrf/dcsvm instead." % model)

    results["FILENAME"] = path.split("/")[-1]
    return results, test_predictions, data['testing_set']['y'], data['testing_set']['t']


if __name__ == '__main__':
    """
    Main program. Call this to run tests.
    Args by examples:
    ?
    """
    try:
        path = sys.argv[1]
    except:
        print("An argument required - (gzipped) pickle file!")
        print("Optional arg 1: model name (cvtlr==cvt/cvtrf/dclr==dc/dcrf/dcsvm)")
        print("""Optional arg 2: k_min,k_max,k_step (comma separated)
                 or baseline (=-1,0,1) or nounder (=0,1,1)""")
        print("""Values <0 (e.g. baseline) merge training&validation,
                 ==0 means: no undersampling""")
        sys.exit(-1)
    print("data=%s" % path)

    try:
        model = sys.argv[2]
    except:
        model = "dclr"
    print("model=%s" % model)

    ranges_suffix = ""
    try:
        if sys.argv[3].lower() == "baseline": 
            ranges = [-1, 0, 1]
        elif sys.argv[3].lower().replace("-","").startswith("nounder"): 
            ranges = [0, 1, 1]
        else:
            ranges = list(map(int, sys.argv[3].split(",")))
        ranges_suffix = "_" + sys.argv[3]
    except:
        ranges = [1, 50, 1]
    k_min,  k_max, k_step = ranges
    print("k_min=%i, k_max=%i, k_step=%i" % (k_min,  k_max, k_step)) 

    if (k_min < 0 and k_max > 0):
        raise ValueError("""
            When merging training&validation (i.e., k<0), ensure that the final evaluation 
            on testing_set is also with the same model (=no other k can be tested at the same time)!""")

    ####################################

    results, test_predictions, test_y, test_t = run_experiment(path, model, k_min,  k_max, k_step)

    ####################################
    # Store results:

    outpath = path.replace(".gz", "").replace(".pickle", "")+"_kselection_"+model+ranges_suffix

    print("Storing results to %s.csv" % outpath)
    results.to_csv(outpath+".csv", index=False)

    print("Storing test set predictions to %s_test_predictions.csv" % outpath)
    df = pd.DataFrame()
    df["test_predictions"] = test_predictions
    df["test_y"] = np.array(test_y, dtype='int')
    df["test_t"] = np.array(test_t, dtype='int')
    df["model"] = model
    df["kvals"] = ranges_suffix[1: ]
    df.to_csv(outpath+"_test_predictions.csv", index=False)
