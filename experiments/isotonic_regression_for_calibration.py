"""
Experiments with isotonic regression for uplift modeling.
"""

import gzip
import pickle
import sys
import numpy as np
import pandas as pd

import models.isotonic_regression as isotonic_regression
import data.load_data as load_data
import metrics.uplift_metrics as uplift_metrics
from models.misc_models import *
from data.data_aux import *


def cvt_experiment(data, k, cvt_class=ClassVariableTransformation, model_id="cvt",
                   test_name=None,
                   file_name=None):
    """
    Class-variable transformation with calibration using isotonic regression.
    Args:
    data (load_data.DatasetCollection): Data set
    k (float): Undersampling factor to use (k * p(y=1) -> p_{new}(y=1)).
    cvt_class (class): Uplift model to use for modeling
    model_id (str): Textual description of model. Used for printing and
     storing results.
    test_name (str): Name for test to be saved in csv with results..
    file_name (str): Name of dataset file
    """

    print("Runinng cvt_experiment with cvt_class=%s model_id=%s k=%s" % (cvt_class, model_id, k))

    results = []

    testing_set = data['testing_set']
    validation_set = data['validation_set']
    training_set = get_training_set(data, k, '11')

    model = cvt_class()
    model.fit(X=training_set['X'], z=training_set['z'])

    # Estimate testing set performance
    testing_set_pred = model.predict_uplift(X=testing_set['X'])
    # Undo effect of change in samling rate:
    testing_set_pred = testing_set_pred / k
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set before IR",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(k))
    results.append([model_id, "testing", k] + metrics2row(metrics))

    print("Testing set metrics without calibration")
    print(metrics)
    
    # Calibrate the results:
    validation_set_pred = model.predict_uplift(X=validation_set['X'])
    validation_set_pred = validation_set_pred / k
    metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                           validation_set_pred,
                                           validation_set['t'])
    results.append([model_id, "validation", k] + metrics2row(metrics))

    ir_model = isotonic_regression.UpliftIsotonicRegression()
    ir_model.fit(validation_set_pred, validation_set['y'], validation_set['t'])

    ir_pred = ir_model.predict_uplift(testing_set_pred)
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           ir_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set after IR",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(k))
    results.append([model_id, "testing_ir", k] + metrics2row(metrics))

    print("Testing set metrics with calibration by isotonic regression")
    print(metrics)

    results = pd.DataFrame(results).rename(
        columns=dict(enumerate(["MODEL", "SUBSET", "k"] + COLNAMES)))
    return results, ir_pred


def dc_experiment(data, k, dc_class=DCLogisticRegression, model_id="dc",
                  test_name=None,
                  file_name=None):
    """
    Double-classifier approach with isotonic regression for calibration.
    Args:
    data (load_data.DatasetCollection): Data set
    k (float): Undersampling factor to use (k * p(y=1) -> p_{new}(y=1)).
    dc_class (class): Uplift model to use for modeling
    model_id (str): Textual description of model. Used for printing and
     storing results.
    test_name (str): Name for test to be saved in csv with results..
    file_name (str): Name of dataset file
    """

    print("Runinng dc_experiment with dc_class=%s model_id=%s k=%s" % (dc_class, model_id, k))

    results = []

    # The code for changes in sampling rate is in load_data
    testing_set = data['testing_set']
    validation_set = data['validation_set']
    training_set = get_training_set(data, k, 'natural')

    model = dc_class()
    # Separate treatment and control samples
    X_t = training_set['X'][training_set['t'], :]
    y_t = training_set['y'][training_set['t']]
    X_c = training_set['X'][~training_set['t'], :]
    y_c = training_set['y'][~training_set['t']]
    model.fit(X_c, y_c, X_t, y_t)

    # Estimate testing set metrics:
    testing_set_pred = model.predict_uplift(testing_set['X'])
    testing_set_pred = testing_set_pred / k
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set before IR",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(k))
    results.append([model_id, "testing", k] + metrics2row(metrics))

    print("Testing set metrics with calibration by isotonic regression")
    print(metrics)

    # Calibrate the results:
    validation_set_pred = model.predict_uplift(X=validation_set['X'])
    validation_set_pred = validation_set_pred / k
    metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                           validation_set_pred,
                                           validation_set['t'])
    results.append([model_id, "validation", k] + metrics2row(metrics))

    ir_model = isotonic_regression.UpliftIsotonicRegression()
    ir_model.fit(validation_set_pred, validation_set['y'], validation_set['t'])

    ir_pred = ir_model.predict_uplift(testing_set_pred)
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           ir_pred,
                                           testing_set['t'],
                                           test_name=test_name,
                                           test_description="testing set after IR",
                                           algorithm=model_id,
                                           dataset=file_name,
                                           parameters='k={}'.format(k))
    results.append([model_id, "testing_ir", k] + metrics2row(metrics))

    print("Testing set metrics with calibration by isotonic regression")
    print(metrics)

    results = pd.DataFrame(results).rename(
        columns=dict(enumerate(["MODEL", "SUBSET", "k"] + COLNAMES)))
    return results, ir_pred


def run_experiment(path, model, k):
    """
    Returns a pandas data frame with metrics.
    Args:
    path (str): Path and filename for dataset
    model (str):
     'cvt' for class-variable transformation with logistic regression
     'cvtrf' for class-variable transformation with random forest
     'dc' or 'dclr' for double-classifier with logistic regression
     'dcrf' for double-classifier with random forest
    k (int): undersampling factor to use
    """

    fp = gzip.open(path, 'rb') if path.endswith(".gz") else open(path, 'rb')
    data = pickle.load(fp)
    fp.close()

    if model == "cvt":
        results, test_predictions = cvt_experiment(data, k, ClassVariableTransformation, "cvt-calibrated",
                                                   test_name="Class-variable transformation with LR, undersampling and IR",
                                                   file_name=path)
    elif model == "cvtrf":
        results, test_predictions = cvt_experiment(data, k, CVTRandomForest, "cvtrf-calibrated",
                                                   test_name="Class-variable transformation with RF, undersampling and IR",
                                                   file_name=path)
    elif model == "dc" or model == "dclr":
        results, test_predictions = dc_experiment(data, k, DCLogisticRegression, "dc-calibrated",
                                                  test_name="Double-classifier with LR, undersampling and IR",
                                                  file_name=path)
    elif model == "dcrf":
        results, test_predictions = dc_experiment(data, k, DCRandomForest, "dcrf-calibrated",
                                                  test_name="Double-classifier with RF, undersampling and IR",
                                                  file_name=path)
    else:
        raise ValueError("Unknown model name=%s! Try: cvt/cvtrf/dclr==dc/dcrf/dcsvm instead." % model)

    results["FILENAME"] = path.split("/")[-1]
    return results, test_predictions, data['testing_set']['y'], data['testing_set']['t']



if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except:
        print("An argument required - (gzipped) pickle file!")
        print("Optional arg 1: model name (cvtlr==cvt/cvtrf/dclr==dc/dcrf/dcsvm)")
        print("Optional arg 2: k-value")
        sys.exit(-1)
    print("data=%s" % path)

    try:
        model = sys.argv[2]
    except:
        model = "dclr"
    print("model=%s" % model)


    try:
        k = int(sys.argv[3])
    except:
        k = 0
    print("k=%s" % k)

    results, test_predictions, test_y, test_t = run_experiment(path, model, k)

    outpath = path.replace(".gz", "").replace(".pickle", "") + "_calibration_"+model + ("_k%s" % k)

    print("Storing results to %s.csv" % outpath)
    results.to_csv(outpath + ".csv", index=False)

    print("Storing IR-calibrated test set predictions to %s_test_ir_predictions.csv" % outpath)
    df = pd.DataFrame()
    df["test_predictions"] = test_predictions
    df["test_y"] = np.array(test_y, dtype='int')
    df["test_t"] = np.array(test_t, dtype='int')
    df["model"] = model
    df["kvals"] = k
    df.to_csv(outpath+"_test_ir_predictions.csv", index=False)