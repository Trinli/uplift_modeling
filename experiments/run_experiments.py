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

import models.data_shared_lasso as data_shared_lasso  # Original benchmark
import models.double_classifier as double_classifier  # Basic approach
import models.class_variable_transformation as class_variable_transformation  # CVT
import data.revert_label_transformation as revert_label_transformation  # Athey's version of CVT
import models.residual_model as residual_model
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

#DATASETS = [HILLSTROM_FORMAT, CRITEO_FORMAT]



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
    tests = [{'test_name': 'Origial DSL',
              'test_description': 'Data-shared lasso (OLS + L1',
              'type': 'DSL',
              'base_learner': 'LASSO'},

             {'test_name': 'DC: LR',
              'test_description': 'Double-classifier with logistic regression',
              'type': 'DC',
              'base_learner': 'LR'},

             {'test_name': 'DC: SVM + IR',
              'test_description': 'Double-classifier with support vector machine and isotonic regression',
              'type': 'DC',
              'base_learner': 'SVM'
             },

             {'test_name': 'CVT: LR',
              'test_description': 'Class-variable transformation with logistic regression',
              'type': 'CVT',
              'base_learner': 'LR'},

             {'test_name': 'CVT: LR-w',
              'test_description': 'Class-variable transformation with logistic regression and re-weighting of one treatment group',
              'type': 'CVT',
              'base_learner': 'LR-w'},

             {'test_name': 'RL: OLS',
              'test_description': 'Rever-label transformation with linear regression',
              'type': 'RL',
              'base_learner': 'OLS'},

             # Perhaps add residual model here.
             {'test_name': 'RM: OLS',
              'test_description': 'Residual model with linear regression',
              'type': 'RM',
              'base_learner': 'OLS'}
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
                if test['base_learner'] == 'LASSO':
                    # The original Data-shared Lasso!
                    tmp_model = data_shared_lasso.DataSharedLasso()
                else:
                    # If we are not runnign lasso, but the test
                    # specifies "DSL"
                    break

            if test['type'] == 'DC':
                # Double-classifier approach
                print("Base learner: {}".format(test['base_learner']))
                if test['base_learner'] == 'LR':
                    tmp_model = double_classifier.LogisticRegression()
                elif test['base_learner'] == 'SVM':
                    tmp_model = double_classifier.SupportVectorMachine()

            elif test['type'] == 'CVT':
                if test['base_learner'] == 'LR':
                    tmp_model = class_variable_transformation.LogisticRegression()
                elif test['base_learner'] == 'LR-w':
                    # Logistic with reweighting of some samples
                    # This is what Diemert & al used in 2018
                    tmp_model = class_variable_transformation.LogisticReWeight()

            elif test['type'] == 'RL':
                if test['base_learner'] == 'OLS':
                    tmp_model = revert_label_transformation.RLRegression()
                else:
                    break

            elif test['type'] == 'RM':
                if test['base_learner'] == 'OLS':
                    tmp_model = residual_model.ResidualLinearRegression()
                else:
                    break

            print("Model set")

            # Run all tests with all undersampling options:
            for sampling_rate in ['natural', '11', '1111']:
                if sampling_rate == 'natural':
                    print("Training...")
                    tmp_model.fit(data)
                    print("Done.")
                else:
                    try:
                        print("Training...")
                        print(sampling_rate)
                        tmp_model.fit(data, undersampling=sampling_rate)
                        print("Done.")
                    except TypeError:
                        print("Undersampling not defined for current model.")
                        print("Moving on to next...")
                        # If the model does not support undersampling, skip
                        # rest and go on with next model.
                        break

                # Predict stage:
                print("Predicting...")
                predictions = tmp_model.predict_uplift(data['testing_set']['X'])
                print("Estimating metrics...")
                metrics = uplift_metrics.UpliftMetrics(
                    data['testing_set']['y'],
                    predictions,
                    data['testing_set']['t'],
                    test_name=test['test_name'],
                    test_description=test['test_description'],
                    algorithm=test['type'] + " with " + test['base_learner'],
                    dataset=dataset['file_name'] + " with sampling rate " + str(sampling_rate),
                    parameters=None
                )
                print(metrics)
                # Write results to file:
                metrics.write_to_csv()


if __name__ == '__main__':
    # If this is the main program, run it!
    main()
