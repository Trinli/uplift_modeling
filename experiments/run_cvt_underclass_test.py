"""
Script to test hypotheses concerning changes in sampling rates

I have modified this heavily for use in extended_undersampling_experiments.py.
"""
import copy
import metrics.uplift_metrics as uplift_metrics
import numpy as np
import pandas as pd
import pickle
import sys
import gzip
import time
import gc
from models.misc_models import *
from experiment_aux import *


def cvt_experiment(data, k_min=-1, k_max=500, k_step=10, 
                   cvt_class=ClassVariableTransformation, model_id="cvt",
                   data_info='',
                   metrics_file='datasets/criteo2/generated_subsamples/uplift_results.csv'):
    """
    CVT experiment that figures out optimal k with validation set metrics.

    Args:
    data (load_data.DatasetCollection): Object with training, validation, and testin set.
    k_min (int): Minimum k to test. -1 causes training and validation sets to be merged.
    k_max (int): Maximum k to test.
    k_step (int): Step size for k. (k_max-k_min)/k_step ends up being the resolution.

    Returns:
    List of optimal model testing set metrics and related k. Also writes said metrics to metric_file.

    Notes:
    k_* could be changed to floats. Range* does currently not support that.
    """
    print("Runinng cvt_experiment with cvt_class=%s model_id=%s" % (cvt_class, model_id))

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
        try:
            training_set = get_training_set(data, i, '11')
        except AssertionError:
            print("k {} causes all negative sampels to be droped".format(i))
            break
        model.fit(X=training_set['X'], z=training_set['z'])
        predictions = model.predict_uplift(X=validation_set['X'])
        metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                               predictions,
                                               validation_set['t'])

        if metrics.improvement_to_random > best_improvement_to_random:
            best_improvement_to_random = metrics.improvement_to_random
            best_model = copy.deepcopy(model)
            best_k = i

        print("Evaluated in %.2f seconds" % (time.time()-start))        
        print(metrics)

    # Estimate testing set performance
    testing_set_pred = best_model.predict_uplift(X=testing_set['X'])
    if cvt_class is ClassVariableTransformation:
        algo = 'k-cvt-lr'
        txt = "k-cvt-lr vs. dataset size"
    elif cvt_class is CVTRandomForest:
        algo = 'k-cvt-rf'
        txt = 'k-cvt-rf vs. dataset size'
    else:
        algo = 'k-cvt (-unknown)'
        txt = 'k-cvt (-unknown) vs. dataset size'
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name='k_undersampling_cvt',
                                           test_description=txt,
                                           algorithm=algo,
                                           dataset=data_info,
                                           parameters='k: {}'.format(best_k))
    metrics.write_to_csv(file_=metrics_file)
    print("Testing set metrics for CVT and best k ({}):".format(best_k))
    print(metrics)

    return [metrics, best_k]


def dc_experiment(data, k_min=-1, k_max=50, k_step=1, 
                  dc_class=DCLogisticRegression, model_id="dc",
                  metrics_file='datasets/criteo2/generated_subsamples/uplift_results.csv',
                  data_info=''):
    """
    Function for DC-experiments with k-undersampling that figures out optimal k using
    validation set procedure.

    Args:
    data (load_data.DatasetCollection): Object with training, validation, and testin set.
    k_min (int): Minimum k to test. -1 causes training and validation sets to be merged.
    k_max (int): Maximum k to test.
    k_step (int): Step size for k. (k_max-k_min)/k_step ends up being the resolution.

    Returns:
    List of optimal model testing set metrics and related k.

    Notes:
    k_* could be changed to floats. Range* does currently not support that. Also writes said
    metrics to metrics_file.
    """
    print("Runinng dc_experiment with dc_class=%s model_id=%s" % (dc_class, model_id))

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
        try:
            training_set = get_training_set(data, i, 'natural')
        except AssertionError:
            print("k {} causes # negative samples to drop to zero".format(i))
            break
        # Separate treatment and control samples
        X_t = training_set['X'][training_set['t'], :]
        y_t = training_set['y'][training_set['t']]
        X_c = training_set['X'][~training_set['t'], :]
        y_c = training_set['y'][~training_set['t']]
        model.fit(X_c, y_c, X_t, y_t)

        predictions = model.predict_uplift(validation_set['X'])
        metrics = uplift_metrics.UpliftMetrics(validation_set['y'],
                                               predictions,
                                               validation_set['t'])
        if metrics.improvement_to_random > best_improvement_to_random:
            best_improvement_to_random = metrics.improvement_to_random
            best_model = copy.deepcopy(model)
            best_k = i

        print("Evaluated in %.2f seconds" % (time.time()-start))
        print(metrics)        

    # Estimate testing set metrics:
    testing_set_pred = best_model.predict_uplift(testing_set['X'])
    if dc_class is DCLogisticRegression:
        algo = 'k-dc-lr'
        txt = "k-dc-lr vs. dataset size"
    elif dc_class is DCRandomForest:
        algo = 'k-dc-rf'
        txt = 'k-dc-rf vs. dataset size'
    else:
        algo = 'k-dc (-unknown)'
        txt = 'k-dc (-unknown) vs. dataset size'
    metrics = uplift_metrics.UpliftMetrics(testing_set['y'],
                                           testing_set_pred,
                                           testing_set['t'],
                                           test_name='k_undersampling_dc',  #LR or rf?
                                           test_description=txt,
                                           algorithm=algo,
                                           dataset=data_info,
                                           parameters='k: {}'.format(best_k))

    metrics.write_to_csv(file_=metrics_file)

    print("DC testing set metrics")
    print("best k: {}".format(best_k))
    print(metrics)

    return [metrics, best_k]


def run_experiment(path, model, k_min,  k_max, k_step):
    """ Returns a pandas data frame with metrics for each k in range. """

    fp = gzip.open(path,'rb') if path.endswith(".gz") else open(path,'rb')
    data = pickle.load(fp)
    fp.close()

    if model=="cvt":
        results, test_predictions = cvt_experiment(data, k_min,  k_max, k_step, ClassVariableTransformation, "cvt")
    elif model=="cvtrf":
        results, test_predictions = cvt_experiment(data, k_min,  k_max, k_step, CVTRandomForest, "cvtrf")
    elif model=="dc" or model=="dclr":
        results, test_predictions = dc_experiment(data, k_min,  k_max, k_step, DCLogisticRegression, "dc")
    elif model=="dcrf":
        results, test_predictions = dc_experiment(data, k_min,  k_max, k_step, DCRandomForest, "dcrf")
    elif model=="dcsvm":
        results, test_predictions = dc_experiment(data, k_min,  k_max, k_step, DCSVM, "dcsvm")
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
        if sys.argv[3].lower()=="baseline": 
            ranges = [-1, 0, 1]
        elif sys.argv[3].lower().replace("-","").startswith("nounder"): 
            ranges = [0, 1, 1]
        else:
            ranges = list(map(int,sys.argv[3].split(",")))
        ranges_suffix = "_"+sys.argv[3]
    except:
        ranges = [1, 50, 1]
    k_min,  k_max, k_step = ranges
    print("k_min=%i, k_max=%i, k_step=%i" % (k_min,  k_max, k_step)) 

    if (k_min<0 and k_max>0):
        raise ValueError("""
            When merging training&validation (i.e., k<0), ensure that the final evaluation 
            on testing_set is also with the same model (=no other k can be tested at the same time)!""")

    ####################################

    results, test_predictions, test_y, test_t = run_experiment(path, model, k_min,  k_max, k_step)

    ####################################
    # Store results:

    outpath = path.replace(".gz","").replace(".pickle","")+"_kselection_"+model+ranges_suffix

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
