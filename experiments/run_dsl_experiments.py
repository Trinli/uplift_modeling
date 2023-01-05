"""
Code for testing different kinds of uplift models with at least
the Criteo-Uplift dataset and the Hillstrom Mine that Data
dataset.

Plan:
1. Load data (data loader done!)
2. Create approach that iterates over datasets in DATASETS and
    models defined in MODELS. The class UpliftMetrics in
    uplift_metrics.py handles metrics.

2. Models to be tested
   -double classifier approach (logistic regression and SVM)
   -neural net with DSL-type penalty, p(y| X, t*X)
   -some decision tree for uplift modeling
   -revert label: reweighting of class (Athey & Imbens)
   -class-variable transformation: Z=1 for (y=1, t=1) and (y=0, t=0)
    (Jaskowski & Jaroszewicz, 2012)
    -requires N_t == N_c (can be enforced, but loss of data)
   -data-shared lasso
3. Collect metrics
4. Tie to other results?


Thoughts
-loop over datasets
--loop over models [list of functions!]
-- how do I combine only some models with e.g. undersampling?
   separate functions for these?
   do I make a list with the functions I want to run? yes!
"""
import data.load_data as load_data  # Dataset handling
import metrics.uplift_metrics as uplift_metrics  # Metrics

import models.uplift_neural_net as uplift_neural_net  # Base class for DSL-neural net
import models.deep_dsl_neural_net as deep_dsl_neural_net  #
import models.skip_neural_net as skip_neural_net  # Neural net with skip-connections
import copy
import data.pickle_dataset as pickle_dataset

# Use a mini-version of data for tests
CRITEO_FORMAT = load_data.DATA_FORMAT
# CRITEO_FORMAT['file_name'] = 'criteo_100k.csv'  # This is for testing.
# CRITEO_FORMAT['file_name'] = 'criteo_uplift.csv'
# Pickled Criteo data files with differing randomization:
criteo_files = ['criteo-uplift.csv155768161.pickle',
                'criteo-uplift.csv1740898770.pickle',
                'criteo-uplift.csv3147263977.pickle']
HILLSTROM_FORMAT = load_data.HILLSTROM_FORMAT_1
# This defines what datasets the tests should be run on
hillstrom_files = ['Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv1389087885.pickle',
                   'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv2314272100.pickle',
                   'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv718803072.pickle']

# Populate list with all randomized files
DATASETS = []
for data_file in hillstrom_files:
    tmp = copy.deepcopy(HILLSTROM_FORMAT)
    tmp['file_name'] = data_file
    DATASETS.append(tmp)
for data_file in criteo_files:
    tmp = copy.deepcopy(CRITEO_FORMAT)
    tmp['file_name'] = data_file
    DATASETS.append(tmp)


def main():
    """
    Main loop for this script.

    1. Loop over datasets specified
    2. Loop over models specified
       -models
         network variants
       -hyperparameters
         penalties
         architectural decisions (e.g. neural nets)
       -data mangling
         2 versions of undersampling
         class-variable transformation with bot Z and R
    """

    # Define tests to be run:
    tests = [{'test_name': 'NN-DSL',  # Grid search with 10 nets
              'test_description': 'DSL-approach with neural net',
              'type': 'DSL',
              'base_learner': 'NN'},

             {'test_name': 'NN-DEEP-DSL', # Grid search with 10 nets
              'test_description': 'DSL-like approach where both x and t*x first progress in their own networks and penalties are on the weights connecting the nodes from the t*x network',
              'type': 'DSL',
              'base_learner': 'DEEP-NN'},

             # This takes like 100 times longer than any of the other.
             # Run separately.
             {'test_name': 'RANDOM-NN',  # Grid search with 400 nets
              'test_description': 'Random search of deep DSL neural network',
              'type': 'DSL',
              'base_learner': 'RAND-NN'},

             {'test_name': 'NN-SKIP',  # Grid-search with like 10 different nets
              'test_description': "Approach with only [x, t] as features, but the 't' is also served to the inner layers with skip-connections",
              'type': 'NN-SKIP',
              'base_learner': 'NN'}
    ]

    for dataset in DATASETS:
        print("Loading data...")
        #data = load_data.DatasetCollection('./datasets/' + dataset['file_name'],
        #                                   dataset)
        # Use pickled-datasets instead of csv for faster and more memory efficient
        # loading:
        data = pickle_dataset.load_pickle(dataset['file_name'])
        print("Done.\n")

        # Iterate through defined tests:
        for test in tests:
            # Set model:
            print("Test type: {}".format(test['type']))
            if test['type'] == 'DSL':
                if test['base_learner'] == 'NN':
                    # This part does both fit and model selection:
                    tmp_model = uplift_neural_net.DSLNeuralNet.init_grid_search(
                        data=data,
                        n_hidden_units=[128],
                        dropout_rates=[.5],
                        dsl_penalties=[.1, .03, .01, .003, .001, .0003, .0001,
                                       .00003, .00001])

                elif test['base_learner'] == 'DEEP-NN':
                    tmp_model = deep_dsl_neural_net\
                                .DeepDslNeuralNet\
                                .init_grid_search_neural_net(
                                    data=data,
                                    dsl_penalties=
                                    [.1, .03, .01, .003, .001, .0003, .0001,
                                     .00003, .00001])

                elif test['base_learner'] == 'RAND-NN':
                    tmp_model = deep_dsl_neural_net\
                                .DeepDslNeuralNet\
                                .init_random_search_neural_net(
                                    data,
                                    n_networks=400)

            elif test['type'] == 'NN-SKIP':
                if test['base_learner'] == 'NN':
                    tmp_model = skip_neural_net.\
                                SkipNeuralNet.\
                                init_grid_search(
                                    data=data,
                                    skip_layers=
                                    [[0], [1], [2], [3],
                                     [0, 1], [0, 1, 2],
                                     [0, 1, 2, 3],
                                     [2, 3], [1, 3], [0, 3]])

            # Predict stage:
            print("Predicting...")
            predictions = tmp_model.predict_uplift(load_data.DatasetWrapper(
                data['testing_set']))
            print("Estimating metrics...")
            metrics = uplift_metrics.UpliftMetrics(
                data['testing_set']['y'],
                predictions,
                data['testing_set']['t'],
                test_name=test['test_name'],
                test_description=test['test_description'],
                algorithm=test['type'] + " with " + test['base_learner'],
                dataset=dataset['file_name'] + " with sampling rate natural",
                parameters=None
            )
            print(metrics)
            # Write results to file:
            metrics.write_to_csv()


if __name__ == '__main__':
    # If this is the main program, run it!
    main()
