"""
THIS FILE DOES NOT CONTAIN ANYTHING COMPLETE. I THINK...
OR MAYBE IT IS JUST A COPY OF THE EXPERIMENTS WE RAN FOR ACML?

Actual experiments for extended undersampling

Should include:
1. "Basic" comparison of CVT-LR, CVT-RF, DC-LR,
 DC-RF and, most importantly, CRF. This is to
 show that CRF is not actually as good as claimed.
 -Other benchmarks as well?
 The test focuses on different size datasets that
 will simultaneously show how the benefit of
 undersampling is dependent on sample size.
"""
import datetime
import data.load_data as load_data
import experiments.run_cvt_underclass_test as run_cvt_underclass_test
import experiments.run_crf_experiment as run_crf_experiment

METRICS_FILE = 'uplift_results.csv'


def run_experiments():
    """
    The first experiments here test for performance of k-undersampling over five different
    uplift classifiers (CVT and DC with LR and RF, CRF) and variable sample size. CRF could
    not be run for the full Criteo-1 dataset, but now we have subsamples the Criteo 2 (with
    fewer samples) to 1:1 for treatment:control, and then further reduce the size of that
    dataset.
    """
    format_ = load_data.CRITEO_FORMAT
    results = []
    today = str(datetime.datetime.now())
    for i in range(7, -1, -1):
        # Inverse order. The smallest dataset is run first.
        format_['file_name'] = 'datasets/criteo2/generated_subsamples/' + str(2**i) + '.csv'
        print("File name: {}".format(format_['file_name']))
        data = load_data.DatasetCollection(format_['file_name'], format_)
        # E.g. cvt_underclass_test takes the entire DatasetCollection as input and
        # figures out k, validation set metrics, and testing set metrics on its own.
        # k_min, k_max, k_step?
        k_min = 1
        k_max = 400  # 401.7 corresponds to 1:1 positive to negative rate in treated samples for criteo2.
        k_step = 10
        # Models are ready in cvt_underclass_test.py!!!
        # Incl. DC-LR, DC-RF, CVT-LR, and CVT-RF!
        data_info = format_['file_name']
        #... = crf_experiment.uplift_rf_experiment()
        # Maybe create a version of uplift_rf_experiment with matching signature? And what do I wo with k's?
        # Currently uplift rf does not handle k-undersampling.
        tmp, test_predictions = run_cvt_underclass_test.cvt_experiment(data, k_min, k_max, k_step,
                                                                   run_cvt_underclass_test.ClassVariableTransformation, "cvt" + today,
                                                                   data_info=data_info,
                                                                   metrics_file=METRICS_FILE)
        results.append(tmp)
        tmp, test_predictions = run_cvt_underclass_test.cvt_experiment(data, k_min, k_max, k_step,
                                                                   run_cvt_underclass_test.CVTRandomForest, "cvtrf" + today,
                                                                   data_info=data_info,
                                                                   metrics_file=METRICS_FILE)
        results.append(tmp)
        tmp, test_predictions = run_cvt_underclass_test.dc_experiment(data, k_min, k_max, k_step,
                                                                  run_cvt_underclass_test.DCLogisticRegression, "dc" + today,
                                                                  data_info=data_info,
                                                                  metrics_file=METRICS_FILE)
        results.append(tmp)
        tmp, test_predictions = run_cvt_underclass_test.dc_experiment(data, k_min, k_max, k_step,
                                                                  run_cvt_underclass_test.DCRandomForest, "dcrf" + today,
                                                                  data_info=data_info,
                                                                  metrics_file=METRICS_FILE)
        results.append(tmp)

        metrics, best_k = run_crf_experiment.k_undersamping_crf(data, k_min, k_max, k_step,
                                                            "crf-k" + today,
                                                            metrics_file=METRICS_FILE)
        results.append(metrics)
        # functions above write results into csv-file.
        # Add CRF. crf_experiments.py in voter_experiments branch!
        # THERE IS NO K_UNDERSAMPLING FOR CRF HERE!!! NEEDS FIX!
        # To run r-stuff on ukko2, first purge modules, then run "env3" (loads also python module)
        # and lastly run 'rmodule'. These will load my virtual environment, the python module, and
        # the R-modules - all needed.
        # tmp = crf_experiment.crf_model(data, str(2**i) + '.csv')

        # There is also a flaw in the older models (cvt and dc) where undersampling sometimes causes
        # subsamples that contain no positive samples (due to the positive samples being in minority).
        # How does this happen? Isn't the subsamling stratified?
        # Maybe k ends up being too large causing the dataset to contain only positive samples. This
        # sounds most plausible.

        # Perhaps also track memory usage. Would be important to claim that grf is inefficient.


if __name__ == '__main__':
    # Run actual program
    run_experiments()
