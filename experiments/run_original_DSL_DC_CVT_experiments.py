"""
Some code to test Criteo dataset.
1. Read in data and print some statistics
2. item[12] is the treatment-label
   -it only indicates whether Criteo's system was turned on,
    not whether the user was actually targeted.
   -item[15] indicates whether the user was actually targeted,
    but cannot be subsetted as there is no reference point then.
3.
Note:
-The uplift model here assumes that Criteo's system was unavailable
for randomly selected users, i.e. that it was not turned off only
during nights or similar. This would guarantee that both treatment
and control samples for valid subsets of the entire population.


The results indicate that there is a healthy 37% increase in conversion rate for
treated samples compared to control samples. Hence the treatment has an effect.
The probability for the treatment group having a larger conversion rate than the
control group is virtually 100%.

I think the setup was such that "treated samples" simply had the ad serving backend
turned on, whereas control samples had it turned off. There is separately item[15]
that tracks whether the user actually got some treatment. There was 31653 samples
that were in the treatment group, that were in fact treated, and that ended up
converting. About 4.1% of all samples in the treatment group actually received
some treatment.

If the dataset was not anonymized already, this migth actually be similar to us
trying to model change in conversion rate if we only put the user in the treatment
group - not if we actually end up treating him.
-A bit weird, and considering that our effect size is considerably smaller, it
might not be adviceable to pursue this with our data.


#Example of rows
>>> data[0]  # Title row
['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'treatment', 'conversion', 'visit', 'exposure']
>>> data[1]  # actual data
['1.9919806210563484', '3.2636406358678913', '9.241052455095232', '3.735870942448979', '3.5067331320159765', '10.161280650613378', '2.981720887063089', '-0.16668935847080157', '-0.5843916204548227', '9.850093453658753', '-1.8608999471198722', '4.157648047401516', '1', '0', '0', '0']


Plan:
1. Subsample data for quicker tests
2. Models to be tested
   -double classifier approach (logistic regression and SVM)
   -neural net with DSL-type penalty, p(y| X, t*X)
   -some decision tree for uplift modeling
   -revert label: reweighting of class
   -class-variable transformation: Z=1 for (y=1, t=1) and (y=0, t=0)
    -(Athey and Imbens, 2016) or (Jaskowski & Jaroszewicz)?
    -requires N_t == N_c (can be enforced, but loss of data)
   -data-shared lasso
3. Collect metrics
4. Tie to other results?
"""


import copy
import pytest
import numpy as np
# import torch
import random
from sklearn.svm import SVC
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import generate_neural_net
import data.load_data as load_data
import metrics.uplift_metrics as uplift_metrics
import models.neural_net as neural_net
# import dsl
import dsl_neural_net
import dsl_deep_neural_net
import models.skip_neural_net as skip_neural_net
import skip_cumulative_neural_net
import residual_neural_net
import cvt_neural_net

# Set data type and model to train
DATA_TYPE = 'csv'
MODEL_TYPE = 'generate_neural_net'  # Options below in section 'Model types'
PENALTY = True
PENALTY_LAMBDA = 1
SKIP_LAYER = 3
SKIP_LAYERS = [0, 1, 2, 3]
MODE = 'normal'  # {'test', 'normal'}
base_learner_for_cvt = 'logistic'  # Only used if MODEL_TYPE = 'class_variable_transformation'
# 'nn', 'svm', 'logistic', 'class_variable_transformation', 'nn_dsl', 'nn_skip', 'nn_deep_dsl', 'nn_cumulative_skip'
# 'nn_residual', 'generate_neural_net'

"""
Model types:
'nn': double-classifier approach using neural nets
'svm': double-classifier approach with SVM, radial kernel and hyperparameter selection
(currently the control-model is set to use the same hyperparameters as the ones producing
maximum auc_roc for the treatment-model)
'logistic': double-classifier approach with logistic regression (L2-penalty)
'class_variable_transformation': as described by Jaskowski & Jarosewicz (2012)
'generate_neural_nets': random search approach based on non-linear DSL
*list not comprehensive
"""


def get_statistics(data):
    samples = len(data) - 1  # -1 for header row. This applies to csv data.
    print("Samples: {}".format(samples))
    conversions = sum([int(item[13]) for item in data[1:]])
    print("Conversions: {}".format(conversions))
    print("Conversion rate: {}".format(conversions / samples))
    assert conversions / samples == pytest.approx(.00229404), "Conversion rate does not match"
    samples_treatment = sum([int(item[12]) for item in data[1:]])
    samples_control = samples - samples_treatment
    conversions_treatment = sum([int(item[12]) * int(item[13]) for item in data[1:]])
    conversions_control = conversions - conversions_treatment
    print("Treated samples: {} ({}% of all)".format(samples_treatment, samples_treatment / samples * 100))
    print("Control samples: {} ({}% of all)".format(samples_control, samples_control / samples * 100))
    print("Treated conversions: {}".format(conversions_treatment))
    print("Control conversions: {}".format(conversions_control))
    print("Conversion rate for treated samples: {}".format(conversions_treatment / samples_treatment))
    print("Conversion rate for control samples: {}".format(conversions_control / samples_control))
    print("Effect size of treatment: {}".format((conversions_treatment / samples_treatment) /
                                                (conversions_control / samples_control)))
    potentially_affected = sum([int(item[12]) * int(item[13]) * int(item[15]) for item in data[1:]])
    print("There were {} potentially positively affected samples (treatment + conversion).".format(potentially_affected))
    actual_samples_treatment = sum([int(item[15]) for item in data[1:]])
    print("Actually treated samples: {}".format(actual_samples_treatment))
    print("This is {}% of the treatment samples.".format(actual_samples_treatment / samples_treatment * 100))
    return


def train_model(data, model_type=MODEL_TYPE):
    training_set = data['training_set']
    validation_set = data['validation_set']
    testing_set = data['testing_set']

    # Dataset-wrapper:
    training_dataset = load_data.CriteoDataset(training_set)

    # Treatment dataset:
    training_dataset.set_mode('treatment')

    # Control dataset:
    training_dataset.set_mode('control')

    validation_dataset = load_data.CriteoDataset(validation_set)
    testing_dataset = load_data.CriteoDataset(testing_set)

    #
    conversion_probability = None

    def undersampling(data):
        positive_idx = [bool(item) for item in data[:]['y']]
        negative_idx = [not item for item in positive_idx]
        positive_samples = data[positive_idx]
        negative_samples = data[negative_idx]
        n_positives = int(sum(data[:]['y']))
        assert positive_samples['X'].shape[0] == n_positives, "Error in undersampling."
        # We assume randomized samples and do not shuffle them again.
        # We also assume there are more negative samples than positive ones.
        tmp_negatives = {'X': negative_samples['X'][:n_positives],
                         'y': negative_samples['y'][:n_positives]}
        result = {'X': np.concatenate([positive_samples['X'], tmp_negatives['X']]),
                  'y': np.concatenate([positive_samples['y'], tmp_negatives['y']])}
        return result

    if model_type == 'nn':  # "double-classifier nn"?
        # All code for training neural net
        # Train treatment-model:
        training_dataset.set_mode('treatment')
        validation_dataset.set_mode('treatment')
        treatment_model = neural_net.NeuralNet(training_dataset[:]['X'].shape[1])
        treatment_model.train(training_dataset, validation_dataset)
        # Train control-model:
        training_dataset.set_mode('control')
        validation_dataset.set_mode('control')
        control_model = neural_net.NeuralNet(training_dataset[:]['X'].shape[1])
        control_model.train(training_dataset, validation_dataset)

        # Prediction and evaluation phase:
        # Should use testing set next, not validation set... Oh, well.
        testing_dataset.set_mode('all')
        treatment_scores = treatment_model.predict(testing_dataset).detach().numpy()
        control_scores = control_model.predict(testing_dataset).detach().numpy()
        uplift_for_testing_set = np.array([item_t - item_c for item_t, item_c
                                           in zip(treatment_scores, control_scores)])
        uplift_for_testing_set = np.array([item[0] for item in uplift_for_testing_set])

    elif model_type == 'nn_dsl':  # Data-shared lasso -type of approach with neural nets.
        # All code for training DSL neural net
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create model
        model = dsl_neural_net.NeuralNet(training_dataset[:]['X'].shape[1], penalty=PENALTY, penalty_lambda=PENALTY_LAMBDA)
        model.train(training_dataset, validation_dataset, n_epochs=100)
        # Predict uplifts:
        uplift_for_testing_set = model.predict_uplift(testing_dataset)

    elif model_type == 'nn_skip':  # Data-shared lasso -type of approach with neural nets.
        # All code for training DSL neural net
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create model
        model = skip_neural_net.NeuralNet(training_dataset[:]['X'].shape[1], SKIP_LAYER=SKIP_LAYER)
        model.train(training_dataset, validation_dataset, n_epochs=100)
        # Predict uplifts:
        uplift_for_testing_set = model.predict_uplift(testing_dataset)

    elif model_type == 'dsl':  # Data-shared lasso -type of approach with neural nets.
        # All code for training DSL neural net
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create model
        # I SEEM TO HAVE FORGOTTEN TO ADD DSL TO GIT. REDO.
        model = dsl.grid_search_dsl(training_dataset, validation_dataset)
        # Predict uplifts:
        uplift_for_testing_set = model.predict_uplift(testing_dataset[:]['X'])

    elif model_type == 'nn_cumulative_skip':  # Data-shared lasso -type of approach with neural nets.
        # All code for training DSL neural net
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create model
        model = skip_cumulative_neural_net.NeuralNet(training_dataset[:]['X'].shape[1], SKIP_LAYERS=SKIP_LAYERS)
        model.train(training_dataset, validation_dataset, n_epochs=100)
        # Predict uplifts:
        uplift_for_testing_set = model.predict_uplift(testing_dataset)

    elif model_type == 'nn_deep_dsl':  # Data-shared lasso -type of approach with neural nets.
        # All code for training DSL neural net
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create model
        model = dsl_deep_neural_net.NeuralNet(training_dataset[:]['X'].shape[1], penalty=PENALTY, penalty_lambda=PENALTY_LAMBDA)
        model.train(training_dataset, validation_dataset, n_epochs=100)
        # Predict uplifts:
        uplift_for_testing_set = model.predict_uplift(testing_dataset)

    elif model_type == 'generate_neural_net':  # Data-shared lasso -type of approach with neural nets.
        # Set datasets for dsl-training:
        training_dataset.set_mode('all')
        validation_dataset.set_mode('all')
        # Create new datasets. Note that these are differently split than in other tests.
        # Testing set is identical, though.
        training_dataset_2 = load_data.CriteoDataset(data['training_set_2'])
        # One validation set is used for early stopping, the other for model selection
        validation_dataset_2a = load_data.CriteoDataset(data['validation_set_2a'])
        validation_dataset_2b = load_data.CriteoDataset(data['validation_set_2b'])
        # New signature:
        model = generate_neural_net.random_search_neural_nets(training_dataset_2, validation_dataset_2a, validation_dataset_2b, n_networks=100)
        # Predict uplifts:
        uplift_for_testing_set = model['best_model'].predict_uplift(testing_dataset)

    elif model_type == 'svm':
        # Use default hyperparameters:
        # Also, in the experiment with "set g and c", the hyperparameters were not tuned in the loop!
        models = []
        treatment_model = SVC()
        training_dataset.set_mode('treatment')
        # Use validation data for calibration:
        validation_dataset.set_mode('treatment')
        training_data_tu = undersampling(training_dataset)
        treatment_model.fit(X=training_data_tu['X'], y=training_data_tu['y'])
        val_2_predictions = treatment_model.decision_function(X=validation_dataset[:]['X'])
        tmp_auc_roc = roc_auc_score(validation_dataset[:]['y'], val_2_predictions)
        models.append({'model': treatment_model, 'auc_roc': tmp_auc_roc})
        print("Treatment model auc-roc: {}".format(tmp_auc_roc))
        treatment_ir_model = IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds='clip')
        validation_t_scores = treatment_model.decision_function(X=validation_dataset[:]['X'])
        treatment_ir_model.fit(X=validation_t_scores, y=validation_dataset[:]['y'])

        # Control model
        control_model = SVC()
        training_dataset.set_mode('control')
        validation_dataset.set_mode('control')
        training_data_cu = undersampling(training_dataset)
        control_model.fit(X=training_data_cu['X'], y=training_data_cu['y'])
        val_2_predictions = control_model.decision_function(X=validation_dataset[:]['X'])
        tmp_auc_roc = roc_auc_score(validation_dataset[:]['y'], val_2_predictions)
        # Extract the model that has the highest auc_roc:
        control_ir_model = IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds='clip')
        validation_c_scores = control_model.decision_function(X=validation_dataset[:]['X'])
        control_ir_model.fit(X=validation_c_scores, y=validation_dataset[:]['y'])

        # Prediction phase:
        testing_dataset.set_mode('all')
        tmp = treatment_model.decision_function(X=testing_dataset[:]['X'])
        treatment_prob = treatment_ir_model.predict(tmp)
        tmp2 = control_model.decision_function(X=testing_dataset[:]['X'])
        control_prob = control_ir_model.predict(tmp2)
        uplift_for_testing_set = np.array([item1 - item2 for item1, item2 in zip(treatment_prob, control_prob)])

    elif model_type == 'logistic':
        # Make two standard models
        # LogisticRegression() uses L2-penalty by default.
        # This one now works with dataloader.
        print("DC-logistic is incompatible with the rest of the code as it does not " +
              "return a model that can be used for prediction on testing set.")
        treatment_model = LogisticRegression(solver='lbfgs', penalty='none')
        control_model = LogisticRegression(solver='lbfgs', penalty='none')
        training_dataset.set_mode('treatment')
        treatment_model.fit(X=training_dataset[:]['X'], y=training_dataset[:]['y'])
        training_dataset.set_mode('control')
        control_model.fit(X=training_dataset[:]['X'], y=training_dataset[:]['y'])

        # Predict the difference in conversion probability with the double classifier approach:
        # Turns out LogisticRegression() returns an array of size N*2, i.e. one column for
        # both classes. Column 1 is the one we are interested in, i.e. probability for
        # class '1'.
        testing_dataset.set_mode('all')
        treatment_prob = treatment_model.predict_proba(X=testing_dataset[:]['X'])
        control_prob = control_model.predict_proba(X=testing_dataset[:]['X'])
        uplift_for_testing_set = np.array([item_2 - item_1 for item_1, item_2 in zip(control_prob[:, 1], treatment_prob[:, 1])])
        conversion_probability = control_prob

    elif model_type == 'nn_residual':
        # Model 1: Behavior in control group.
        # Model 2: Difference between control model and what is seen in treatment group.
        training_dataset.set_mode('control')
        validation_dataset.set_mode('control')
        control_model = neural_net.NeuralNet(training_dataset[:]['X'].shape[1])
        control_model.train(training_dataset, validation_dataset, n_epochs=2)

        # Next, model difference:
        training_dataset.set_mode('treatment')
        validation_dataset.set_mode('treatment')
        # Predict for the treatment group using the control model:
        training_predictions = control_model.predict(training_dataset).detach().numpy()
        validation_predictions = control_model.predict(validation_dataset).detach().numpy()
        # Use neural net that takes dataset (with X, y, and t) and y_control
        # Perhaps make a copy of the datasets and adjust y?
        training_residual = copy.deepcopy(training_dataset)
        training_residual.set_y(training_dataset[:]['y'] - np.resize(training_predictions,
                                                                     (training_predictions.shape[0], )))
        validation_residual = copy.deepcopy(validation_dataset)
        validation_residual.set_y(validation_dataset[:]['y'] - np.resize(validation_predictions,
                                                                         (validation_predictions.shape[0], )))
        # Next model the residual. 'y' now falls in [-1, 1]. Binary cross-entropy does not necessarily work?
        # Neither does the sigmoid as such to define the output.
        model = residual_neural_net.NeuralNet(training_dataset[:]['X'].shape[1])
        model.train(training_residual, validation_residual, n_epochs=2)

        # Predict uplift:
        testing_dataset.set_mode('all')
        uplift_for_testing_set = model.predict(testing_dataset).detach().numpy()
        uplift_for_testing_set = np.array([item[0] for item in uplift_for_testing_set])

    elif model_type == 'class_variable_transformation':
        # First, make sure p(t=0) = p(t=1), i.e. that there are equally many treatment and control samples
        # in the dataset.

        # Undersampling of both group and class in one.
        # Guarantees equal group sizes.
        training_dataset.set_mode('treatment')
        training_treatment_n = training_dataset[:]['X'].shape[0]
        training_treatment_positive = sum(training_dataset[:]['y'])
        training_dataset.set_mode('control')
        training_control_n = training_dataset[:]['X'].shape[0]
        training_control_positive = sum(training_dataset[:]['y'])
        tmp_n = int(min(training_treatment_positive, (training_treatment_n - training_treatment_positive),
                        training_control_positive, (training_control_n - training_control_positive)))
        # Resample datasets so that they contain tmp_n positives and tmp_n negatives in both
        # treatment and control groups:
        training_dataset.set_mode('treatment')
        positive_treatment_idx = [bool(item) for item in training_dataset[:]['y']]
        negative_treatment_idx = [not item for item in positive_treatment_idx]
        positive_treatment_idx = [i for i, item in enumerate(positive_treatment_idx) if item is True]
        negative_treatment_idx = [i for i, item in enumerate(negative_treatment_idx) if item is True]
        positive_treatment_idx = random.sample(positive_treatment_idx, tmp_n)
        negative_treatment_idx = random.sample(negative_treatment_idx, tmp_n)
        treatment_idx = positive_treatment_idx + negative_treatment_idx
        training_treatment = {'X': copy.deepcopy(training_dataset[treatment_idx]['X']),
                              'y': copy.deepcopy(training_dataset[treatment_idx]['y']),
                              't': copy.deepcopy(training_dataset[treatment_idx]['t'])}

        training_dataset.set_mode('control')
        positive_control_idx = [bool(item) for item in training_dataset[:]['y']]
        negative_control_idx = [not item for item in positive_control_idx]
        positive_control_idx = [i for i, item in enumerate(positive_control_idx) if item is True]
        negative_control_idx = [i for i, item in enumerate(negative_control_idx) if item is True]
        positive_control_idx = random.sample(positive_control_idx, tmp_n)
        negative_control_idx = random.sample(negative_control_idx, tmp_n)
        control_idx = positive_control_idx + negative_control_idx
        training_control = {'X': copy.deepcopy(training_dataset[control_idx]['X']),
                            'y': copy.deepcopy(training_dataset[control_idx]['y']),
                            't': copy.deepcopy(training_dataset[control_idx]['t'])}

        # Order of samples already randomized.
        print("Treatment size: {}, Control size: {}".format(training_treatment['X'].shape[0],
                                                            training_control['X'].shape[0]))
        training_data_X = np.concatenate([training_treatment['X'], training_control['X']])
        training_data_y = np.concatenate([training_treatment['y'], training_control['y']])
        training_data_t = np.concatenate([training_treatment['t'], training_control['t']])

        # Change class variable

        def class_variable_transformation(a, b):
            # 'a' is class, 'b' is group ('1' implies treatment group)
            if a == b:
                # Case where a == b == 1 of a == b == 0
                return(1)
            elif a != 0 and a != 1:
                print("a: {}".format(a))
            elif b != 0 and b != 1:
                print("b: {}".format(b))
            else:
                return(0)

        # Create class variable transformation:
        training_data_z = np.array([class_variable_transformation(item_y, item_t)
                                    for item_y, item_t in zip(training_data_y, training_data_t)])
        if base_learner_for_cvt == 'logistic':  # Class-variable transformation with logistic regression:
            # Implement this one with logistic regression.
            model = LogisticRegression(penalty='none', solver='lbfgs')
            model.fit(X=training_data_X, y=training_data_z)
            tmp = model.predict_proba(X=testing_dataset[:]['X'])
            print("Unique predictions: {}".format(len(np.unique(tmp))))
            uplift_for_testing_set = np.array([2 * item - 1 for item in tmp[:, 1]])

        elif base_learner_for_cvt == 'nn':  # Neural net version of class-variable transformation
            # Create new dataloader? We have training_data_X, training_data_y
            # training_data_t, and training_data_z. Replace y by z in a numpy
            # array. Make dataloader.
            print("This part (class-variable transformation with neural network) " +
                  "might require changes in dataloader for other datasets than Criteo.")
            training_np_array = np.zeros((training_data_X.shape[0], 16)).astype(np.float32)
            training_np_array[:, :12] = training_data_X
            training_np_array[:, 13] = training_data_z
            tmp_dataset = load_data.CriteoDataset(training_np_array)

            model = cvt_neural_net.NeuralNet(tmp_dataset[:]['X'].shape[1])
            # DOES USING validation_dataset HERE AS IS MAKE SENSE? YES.
            # The metric being maximized is AUUC.
            model.train(tmp_dataset, validation_dataset, n_epochs=100)

            uplift_for_testing_set = model.predict_uplift(testing_dataset)
        else:
            print("No base learner chosen for class-variable transformation.")
            import sys
            sys.exit(0)

    else:
        print("Invalid model type '{}'.".format(MODEL_TYPE))
        import sys
        sys.exit(0)

    if conversion_probability is not None:
        return {'uplift': uplift_for_testing_set,
                'conversion_probability': conversion_probability}  # ,
                # 'model': model}
    else:
        return {'uplift': uplift_for_testing_set,
                'model': model}


def estimate_uplift(dataset,
                    uplift_for_dataset,
                    conversion_probability=None):
    # testing_set as CriteoDataset

    def t_c(item):
        warnings.warn("This format for data_group np.array is not supported anymore!")
        # Format data for uplift metrics
        if item == 1:
            return 'treatment'
        elif item == 0:
            return 'control'
        raise ValueError("Input not allowed.")

    dataset_group = np.array([t_c(item) for item in dataset[:]['t']])

    # Estimate expected calibration error:
    # The function will print the metrics. Not using ece anywhere else.
    # k=100 might be quite a lot. It implies that there will be
    # 1% of the samples in a bin, i.e. out of 6,327,371 we have around
    # 63,274, and in the control group 15% of these, i.e. 9491 samples,
    # which of roughly 0.2% are positive, i.e. 19 positive samples in
    # the smallest group. This is very little. I.e. even with a dataset
    # the size of Criteo's, it is hard to estimate ECE. Perhaps k=10 would
    # be appropriate.

    if conversion_probability is not None:
        # This part is currently implemented only for the double-classifier approach
        # using logistic regression. It was mostly implemented as a sanity check.
        # Plot conversion probability vs. uplift:
        tmp_conversion_probability = np.array([item[1] for item in conversion_probability])
        uplift_metrics.plot_base_probability_vs_uplift(data_probability=tmp_conversion_probability,
                                                       data_uplift=uplift_for_dataset, k=10000)
        uplift_metrics.plot_uplift_vs_base_probability(data_probability=tmp_conversion_probability,
                                                       data_uplift=uplift_for_dataset, k=10000)

    # Estimate expected calibration error:
    ece = uplift_metrics.expected_calibration_error_for_uplift(data_class=dataset[:]['y'],
                                                               data_prob=uplift_for_dataset,
                                                               data_group=dataset_group,
                                                               verbose=True,
                                                               k=10)

    n_samples = len(uplift_for_dataset)
    k = sum(uplift_for_dataset > 0)
    if k == n_samples:
        k = k - 1  # Quick hack. Should perhaps look into 'uplift_metrics.expected_conversion_rates()'.
        # The function produces one too few items, I think.
    print("k: {}".format(k))
    print("N: {}".format(n_samples))
    # Track time
    import time
    t_1 = time.time()
    conversion_list = uplift_metrics.expected_conversion_rates(data_class=dataset[:]['y'],
                                                               data_score=uplift_for_dataset,
                                                               data_group=dataset_group)
    t_2 = time.time()
    print("Elapsed time [s]: {}".format(t_2 - t_1))
    conversion_0 = conversion_list[0]
    conversion_1 = conversion_list[-1]
    conversion_for_treatment_plan = conversion_list[k]
    if(False):
        conversion_for_treatment_plan = uplift_metrics.conversion_k(data_class=dataset[:]['y'],
                                                                    data_score=uplift_for_dataset,
                                                                    data_group=dataset_group, k=k,
                                                                    smoothing=0)
    conversion_random = (n_samples - k) / n_samples * conversion_0 +\
                        k / n_samples * conversion_1
    print("Improvements: {:.3}% to random, {:.3}% to no treatments, {:.3}% to all treatments".format(
        (conversion_for_treatment_plan / conversion_random - 1) * 100,
        (conversion_for_treatment_plan / conversion_0 - 1) * 100,
        (conversion_for_treatment_plan / conversion_1 - 1) * 100))

    print("k: {}, N: {}".format(k, n_samples))
    print("Conversion rate at k=0: {}".format(conversion_0))
    print("Conversion rate at k=N: {}".format(conversion_1))
    print("Conversion rate at k={}: {}".format(k, conversion_for_treatment_plan))
    print("Max(E(conversion rate)): {}".format(max(conversion_list)))
    print("Max conversion at k: {}".format(conversion_list.index(max(conversion_list))))
    print("E_r(conversion rate| random): {}".format(np.mean([conversion_0, conversion_1])))
    print("E_r(conversion rate| plan): {}".format(np.mean(conversion_list)))
    print("Unique scores {} of {} samples.".format(len(np.unique(uplift_for_dataset)), dataset[:]['X'].shape[0]))
    qini_coefficient = uplift_metrics.qini_coefficient(data_class=dataset[:]['y'],
                                                       data_score=uplift_for_dataset,
                                                       data_group=dataset_group)
    print("Qini-coefficient: {}".format(qini_coefficient))

    print("Plotting...")
    uplift_metrics.plot_conversion_rates(data_class=dataset[:]['y'],
                                         data_score=uplift_for_dataset,
                                         data_group=dataset_group,
                                         file_name='criteo_conversion.png')
    if(False):
        uplift_metrics.plot_uplift_curve(data_class=dataset[:]['y'],
                                         data_score=uplift_for_dataset,
                                         data_group=dataset_group,
                                         file_name='criteo_uplift_curve.png')


if __name__ == '__main__':
    print("Model type: {}".format(MODEL_TYPE))
    print("Reading in data...")
    all_data = load_data.load_data(data_type=DATA_TYPE, mode=MODE)
    # data is a dict with training, testing and validation sets separately
    data = load_data.normalize_data(all_data)
    data = load_data.split_data(all_data)
    print("Done.")
    print("Modeling...")  # This is the beef
    tmp = train_model(data, model_type=MODEL_TYPE)  # Function returns uplift for testing samples
    print("Done.")
    print("Estimating metrics...")
    testing_set = data['testing_set']
    testing_dataset = load_data.CriteoDataset(testing_set)
    if 'conversion_probability' in tmp:
        # If we have the baseline_probability, do a plot where
        estimate_uplift(testing_dataset, tmp['uplift'], tmp['conversion_probability'])
    else:
        estimate_uplift(testing_dataset, tmp['uplift'])
