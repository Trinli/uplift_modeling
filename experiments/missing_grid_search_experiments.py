"""
File for missing grid search experiments.
k_t and k_c values were determined from validation set performance.
"""
import experiments.run_crf_experiment as crf
import sys

def main(model = 'dc'):
    if model == 'test':
        filename='criteo_100k.csv'
        path='./datasets/'
        model = 'rf'
    else:
        filename = 'criteo2_1to1.csv'
        path = './datasets/criteo2/'
    if model == 'dc':
        ## Best values for criteo2 and DC: k_t 240, k_c 40. Not true.
        # Best values are (k_t, k_c)
        #[110.0, 170.0]
        k_t = 110.0
        k_c = 170.0
        metrics = crf.uplift_dc_experiment(filename, path, k_t, k_c)
        print(metrics)
    elif model == 'rf':
        # Best values for criteo2 k_t 2.0, k_c 12.391575109118264. Not true.
        # Best values are (k_t, k_c):
        # [32.0, 99.13260087294611
        k_t = 32.0
        k_c = 99.13260087294611
        metrics = crf.uplift_rf_experiment(filename, path, k_t, k_c)
        print(metrics)


if __name__ == '__main__':
    model = sys.argv[1]
    main(model)
