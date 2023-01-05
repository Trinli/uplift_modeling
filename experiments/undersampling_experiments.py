"""
Main file of extended undersampling experiments. This is post ACML.
Maybe for ECML PKDD (Journal track).

This file takes as inputs the data file (location), the
undersampling scheme, the model, and the correction scheme
and trains a model. It further estimates metrics and writes
them to a file that is also provided as parameter.
"""
import copy
from itertools import product
import numpy as np
import re
import sys
import time

import data.load_data
# Undersampling methods are in data.load_data
# Models
from models.undersampling_models import DoubleClassifier, \
    ClassVariableTransformation, UpliftNeuralNet, UpliftRandomForest
# Correction methods:
from models.uplift_calibration import analytic_correction, div_by_k
from models.isotonic_regression import IsotonicRegression
from models import naive_dc
# Metrics
import metrics.uplift_metrics


start_time = time.time()
def print_verbose(txt, verbose=False):
    global start_time
    if verbose:
        print("Time for last phase: {} s".format(time.time() - start_time))
        print(txt)
    start_time = time.time()


def main(data_format, undersampling, model,
         correction, result_file, k_values=None,
         rate=1.0):
    """
    Training and metrics storing function.
    Many imports will be included here to e.g. not need R or
    pytorch for models that do not use them.

    Args:
    data_format (see load_data.DATA_FORMAT): Format for data in
     csv file to be read.
    undersampling ('str'): Undersampling method. Alternatives
     are 'none', 'k_undersampling', 'naive_undersampling', and 'split_undersampling'
    model ('str'): Model to use for experiment. Options are
     'dc_lr', 'cvt_lr', 'uplift_neural_net', 'uplift_rf'.
    correction ('str'): Correction (calibration) method to
     be used to undo bias introduced by undersampling. Options
     are 'none', 'div_by_k', 'calibration', and 'analytic'.
    result_file ('str'): path to file that results should be
     stored in.
    k_values ([int] or [tuple]): If k_values is none, the experiment
     will cover k_values as defined in the code below. Otherwise it
     will use the k_values provided. k_values can be a list with one
     or multiple items, or a list of lists with two items for
     split-undersampling.
    verbose (bool): Print details along the way.
    rate (float): In [0, 1]. Set to < 1 to subsample training and
     validation sets.
    """
    # Initialize looping variables
    # AUUC should be in [-1, 1], but could hypothetically approach -2.
    best_auuc = -10.0
    # Load data
    print_verbose("Loading dataset")
    dataset = data.load_data.DatasetCollection(data_format['file_name'], data_format=data_format)
    # Subsample data if rate is smaller than 1:
    if rate < 1.0:
        dataset.subsample(rate)
    print_verbose("Done.")
    if k_values is None:
        # If k-values are not provided, use the ones below.
        print_verbose("Setting k-values")
        if undersampling == 'none':
            k_values = [1.0]
        elif undersampling == 'dc_undersampling':
            k_values = [1.0]
        elif undersampling == 'k_undersampling':
            k_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        elif undersampling == 'naive_undersampling':
            k_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        elif undersampling == 'split_undersampling':
            k_values = product([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
            # Make into list
            k_values = [*k_values]

    if correction == 'analytic':
        print_verbose("Estimating correction parameters for analytic correction.")
        # Estimate p_t and p_c for possible analytic correction
        n_pos_t = np.sum(dataset['training_set']['y'][dataset['training_set']['t']])
        n_tot_t = np.sum(dataset['training_set']['t'])
        p_t = n_pos_t / n_tot_t
        n_pos_c = np.sum(dataset['training_set']['y'][~dataset['training_set']['t']])
        n_tot_c = np.sum(~dataset['training_set']['t'])
        p_c = n_pos_c / n_tot_c

    # Format of validation and testing set varies slightly dependent
    # on model. It doesn't require any memory and is quick.
    # Training and validation set metrics is evaluated for every model.
    # Neural nets use the validation set for early stopping, hence comparing
    # the validation set performance of models is not a valid approach.
    for k in k_values:
        # Get training set straight:
        print_verbose("Iterating over k-values")
        try:
            print_verbose("Undersampling with k {}".format(k))
            if isinstance(k, float):
                if k == 1:
                    training_set = dataset['training_set']
                elif undersampling == 'naive_undersampling':
                    training_set = dataset.naive_undersampling(k)
                elif undersampling == 'k_undersampling':
                    training_set = dataset.k_undersampling(k)
                elif undersampling == 'none':
                    training_set = dataset['training_set']
                if correction == 'analytic':
                    # Make k a list of k_t and k_c that are equal.
                    # Enable consistent signatures for DC-lR an
                    # Uplift RF.
                    k = [k, k]
            elif undersampling == 'split_undersampling':
                training_set = dataset.undersampling(k[0], k[1])
            else:
                print("Selected undersmapling scheme: {}".format(undersampling))
                raise Exception("Selected undersampling not valid (training loop).")
        except AssertionError:
            print("k {} too large for dataset. Continuing...".format(k))
            continue

        # Train model. Instantiate new to prevent e.g. neural net
        # from simply continuing training on previous model.
        if model == 'dc_lr':
            tmp_model = DoubleClassifier()
            if correction == 'analytic':
                tmp_model.fit(training_set['X'][training_set['t'], :],
                              training_set['y'][training_set['t']],
                              training_set['X'][~training_set['t'], :],
                              training_set['y'][~training_set['t']],
                              k_t=k[0], k_c=k[1], p_t=p_t, p_c=p_c)
            else:
                tmp_model.fit(training_set['X'][training_set['t'], :],
                              training_set['y'][training_set['t']],
                              training_set['X'][~training_set['t'], :],
                              training_set['y'][~training_set['t']])
            # The function below does analytic correction if variables available.
            predictions = tmp_model.predict_uplift(dataset['validation_set']['X'])
        elif model == 'naive_dc':
            tmp_model = naive_dc.NaiveDoubleClassifier()
            tmp_model.fit(dataset)
            predictions = tmp_model.predict_uplift(dataset['validation_set']['X'])
        elif model == 'cvt_lr':
            tmp_model = ClassVariableTransformation()
            tmp_model.fit(training_set['X'], training_set['z'])
            predictions = tmp_model.predict_uplift(dataset['validation_set']['X'])
            print(predictions.shape)
        elif model == 'uplift_neural_net':
            print_verbose("Initializing neural net")
            tmp_model = UpliftNeuralNet(training_set['X'].shape[1])
            print_verbose("Training neural net")
            tmp_model.fit(training_set, dataset['validation_set'])
            #predictions = tmp_model.predict_uplift(dataset['validation_set']['X'])
            print_verbose("Predicting using neural net")
            predictions = tmp_model.predict_uplift(dataset['validation_set'])
            print_verbose("Predictions done.")
        elif model == 'uplift_rf':
            tmp_model = UpliftRandomForest()
            if correction == 'analytic':
                tmp_model.fit(training_set, k_t=k[0], k_c=k[1], p_t=p_t, p_c=p_c)
            else:
                tmp_model.fit(training_set)
            # Model does analytic correction if parameters available (from fit()).
            predictions = tmp_model.predict_uplift(dataset['validation_set']['X'])
        else:
            raise Exception("Model {} not valid (training loop).".format(model))

        # Correction (/calibration)
        # 'none', 'div_by_k', 'calibration', and 'analytic'.
        print_verbose("Applying correction")
        if correction == 'none':
            corrected_predictions = predictions
        elif correction == 'div_by_k':
            corrected_predictions = \
                div_by_k(predictions, k)
        elif correction == 'isotonic':
            # Calibrating on validation set
            # y_min not > 0 and y_max not < 1 might cause issues if the probabilities
            # are dealt with in a bayesian way further down the line.
            calibration_model = IsotonicRegression(y_min=0.0, y_max=1.0,
                                                   out_of_bounds='clip')
            calibration_model.fit(predictions, dataset['validation_set']['y'])
            corrected_predictions = calibration_model.predict(predictions)
        elif correction == 'analytic':
            if model != 'uplift_rf' and model != 'dc_lr':
                raise Exception("Analytic correction without DC-LR or Uplift RF is not supported.")
                # corrected_predictions = analytic_correction(predictions, k[0], k[1], p_t, p_c)
            elif model == 'uplift_rf' or model == 'dc_lr':
                # Correction already done internally in algorithm
                corrected_predictions = predictions
        else:
            raise Exception("Correction method {} not valid (training loop).".format(correction))

        # Estimate and store metrics
        # If the model is naive_dc, then the k-values are actually inside the model itself.
        if model == 'naive_dc':
            # Extract k-values from model to store in file:
            k = [tmp_model.k_t, tmp_model.k_c]
        print_verbose("Estimating metrics")
        tmp_metrics = metrics.uplift_metrics.UpliftMetrics(
            dataset['validation_set']['y'], corrected_predictions, dataset['validation_set']['t'],
            algorithm=model, parameters=k, dataset=data_format['file_name'],
            test_description=''
        )
        print("k-parameter")
        print(k)
        print(tmp_metrics)
        print_verbose("Storing results to file")
        tmp_metrics.write_to_csv(result_file)

        # If validation set metrics are best for this, keep
        # model and estimate metrics on _testing__set_.
        if tmp_metrics.auuc > best_auuc:
            best_auuc = tmp_metrics.auuc
            best_k = k  # Can we store tuples just fine?
            best_model = copy.deepcopy(tmp_model)  # ALSO NOT STORING IR MODEL HERE.
            if correction == 'isotonic':
                # Store also IR model
                best_calibration_model = copy.deepcopy(calibration_model)

    # Metrics on testing set:
    print_verbose("Estimating testing set metrics. Predicting...")
    if model == 'dc_lr':
        predictions = best_model.predict_uplift(dataset['testing_set']['X'])
    elif model == 'naive_dc':
        predictions = best_model.predict_uplift(dataset['testing_set']['X'])
    elif model == 'cvt_lr':
        predictions = best_model.predict_uplift(dataset['testing_set']['X'])
    elif model == 'uplift_neural_net':
        predictions = best_model.predict_uplift(dataset['testing_set'])
    elif model == 'uplift_rf':
        if correction == 'analytic':
            predictions = best_model.predict_uplift(dataset['testing_set']['X'])
        else:
            predictions = best_model.predict_uplift(dataset['testing_set']['X'])
    else:
        raise Exception("Model {} not valid (training loop).".format(model))
    # Calibrate testing set predictions:
    print_verbose("Correcting the testing set predictions")
    if correction == 'none':
        corrected_predictions = predictions
    elif correction == 'div_by_k':
        corrected_predictions = \
            div_by_k(predictions, k)
    elif correction == 'isotonic':
        # Calibrating on testing set
        corrected_predictions = best_calibration_model.predict(predictions)
    elif correction == 'analytic' and model != 'uplift_rf':
        corrected_predictions = analytic_correction(predictions, k[0], k[1])
    elif correction == 'analytic' and model == 'uplift_rf':
        # Correction already done internally in algorithm
        corrected_predictions = predictions
    else:
        raise Exception("Correction method {} not valid (training loop).".format(correction))

    print_verbose("Estimating testing set metrics")
    testing_set_metrics = metrics.uplift_metrics.UpliftMetrics(
        dataset['testing_set']['y'],
        corrected_predictions,
        dataset['testing_set']['t'],
        algorithm=model, parameters=best_k, dataset=data_format['file_name'],
        test_description='testing_set_performance'
    )
    print_verbose("Storing testing set results")
    testing_set_filename = result_file + '_testing_set.csv'
    testing_set_metrics.write_to_csv(testing_set_filename)
    # Finally, print estimated testing set metrics.
    print("=" * 40)
    print(testing_set_metrics)
    print_verbose("Done.")


if __name__ == '__main__':
    print("Run experiment as python -m experiments.undersampling_experiments dataset undersampling_scheme " +
          "model correction_method result_file k_values rate")
    parameters = sys.argv
    print(parameters)
    dataset = parameters[1]
    # Check parameters before e.g. loading data? Data loading
    # is slowish.
    if dataset == 'criteo1':
        data_format = data.load_data.CRITEO_FORMAT
        data_format['file_name'] = './datasets/criteo1/criteo1_1to1.csv'
    elif dataset == 'criteo2':
        data_format = data.load_data.CRITEO_FORMAT
        data_format['file_name'] = './datasets/criteo2/criteo2_1to1.csv'
    elif dataset == 'hillstrom':
        data_format = data.load_data.HILLSTROM_CONVERSION
        data_format['file_name'] = './datasets/' + data_format['file_name']
    elif dataset == 'voter':
        data_format = data.load_data.VOTER_FORMAT
        data_format['file_name'] = './datasets/voter/voter_1_to_1_.5_dataset.csv'
    elif dataset == 'zenodo':
        data_format = data.load_data.ZENODO_FORMAT
        data_format['file_name'] = './datasets/zenodo_modified.csv'
    elif dataset == 'starbucks':
        data_format = data.load_data.STARBUCKS_FORMAT
        data_format['file_name'] = './datasets/starbucks.csv'
    elif dataset == 'test':
        # This is for testing purposes
        print("Using a mini set of the Criteo-1 dataset for testing purposes.")
        data_format = data.load_data.CRITEO_FORMAT
        data_format['file_name'] = './datasets/criteo_100k.csv'
    else:
        # Filename for dataset.
        print("Assuming dataset is a criteo dataset")
        data_format = data.load_data.CRITEO_FORMAT
        data_format['file_name'] = dataset

    undersampling = parameters[2]
    assert undersampling in ['none', 'k_undersampling', 
                             'naive_undersampling', 'split_undersampling',
                             'dc_undersampling'], \
                                 "Undersampling {} not a valid selection.".format(undersampling)

    model = parameters[3]
    assert model in ['dc_lr', 'naive_dc', 'cvt_lr', 'uplift_rf', 'uplift_neural_net'], \
        "Model {} not a valid selection.".format(model)

    correction = parameters[4]
    assert correction in ['none', 'div_by_k', 'isotonic', 'analytic'], \
        "Correction method {} not a valid selection.".format(correction)

    # Formulate result_file name based on metadata
    # One combination of dataset, undersampling, model, and correction could
    # end up in on file? That file will then have different k-values in it.
    try:
        result_file = parameters[5]
    except:
        # Actually, this is the default
        result_file = './results/undersampling/result_' + undersampling + '_' + model +\
            '_' + correction + '.csv'
    # Deal with model differences etc in main().

    def floatify(item):
        try:
            tmp = float(item)
            return tmp  # Needs to break here.
        except ValueError:
            # Argh. It is not stored as a list, it is stored as a string,
            # e.g. "'[4.0, 12.39..]'". Regex
            #re... "\[ [\d*]\.[\d*],"
            # ", ... \]"
            pass
        try:
            #p = re.compile('[\d]*\.[\d]*')  # Matches e.g. "41.23"
            p = re.compile('[0-9]+.[0-9]+')
            tmp = p.findall(item)
            if len(tmp) == 0:
                # Previous match did not produce expected results. Try differently:
                q = re.compile('[0-9]+')  # Matches ints.
                tmp = q.findall(item)
            tmp = [float(tmp[0]), float(tmp[1])]
        except IndexError:
            tmp = None
        return tmp

    try:
        # THIS DOES NOT HANDLE SEPARATE k_t AND k_c VALUES?!?
        # Tuple can be passed to main(), i.e. perhaps try to
        # parse it and pass on a list. That is done in some code already. Where?
        # 1. k_values needs to possible to pass as _one_ int (or maybe float)
        # 2. k_values should also be possible to leave empty or set to 'all'
        #  to specify that model should run through all k-values (in main()).
        # 3. k_values should be possible to pass as a tuple or list of _two_
        #  floats.
        k_values = float(parameters[6])
    except IndexError:
        # There was not this many arguments in the program call.
        # Allow program above to decide k-values to run through.
        k_values = None
    except ValueError:
        # E.g. if a string is passed that cannot be transformed into an int (e.g. 'all')
        # Perhaps require k_t and k_c to be passed with decimal point and comma in between
        # without any spaces.
        p = re.compile('[0-9]+.[0-9]+')
        tmp = p.findall(parameters[6])
        if len(tmp) == 0:
            k_values = None
        else:
            k_values = [float(tmp[0]), float(tmp[1])]

    # 'rate' defines how large part of the training and validation set should be used
    # for training. This can be used to test how the models are affected by dataset size.
    try:
        rate = float(parameters[7])
    except IndexError:
        rate = 1.0

    # How can we change this so that k_values 'none' can be passed and in addition a rate after that?
    # Perhaps set a special value for 'k', like 'all', and then set rate? k_values is currently
    # the last parameter passed, so that should be possible.

    if k_values is None:
        main(data_format, undersampling, model, correction, result_file, rate=rate)
    else:
        main(data_format, undersampling, model,
             correction, result_file, [k_values],
             rate=rate)
    # Run something...
