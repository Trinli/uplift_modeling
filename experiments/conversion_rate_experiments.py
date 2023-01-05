"""
Code for conversion rate experiments.
Essentially we want to see how the different methods and undersampling
strategies work when the conversion rate in the groups (treatment and
control) differ. At least k-undersampling should break down as one
assumption (p_t \approx p_c) is violated.

Experiments run on Criteo2 as it should not contain the "bias" Criteo's
guys are talking about.
"""

import models.uplift_calibration as uplift_calibration
import data.load_data as load_data
from itertools import product



CRITEO2_RATES = uplift_calibration.CRITEO2_RATES
# Baseline conversion rate for p_c is 0.0019375880152813366
# Baseline conversion rate for p_t is 0.0030894610674129645
# Start from that. Or maybe drop p_t to 0.0019 too? This will
# satisfy the assumptions of k-undersampling and naive undersampling
# quite well.
pt_natural_rate = CRITEO2_RATES['n_positive_c'] /\
    (CRITEO2_RATES['n_positive_c'] + CRITEO2_RATES['n_negative_c'])
pt_rates = [pt_natural_rate * 2**i for i in range(6)]

datasets = ["criteo2_" + str(rate) + '.csv' for rate in pt_rates]
path = 'datasets/conversion_rate_experiments/'

models = ['dc_lr', 'uplift_rf']
undersampling_schemes = ['none', 'k_undersampling', 'split_undersampling']

def generate_data():
    # CRITEO2_RATES = {
    #     # Criteo published a second dataset called
    #     # criteo-research-uplift-v2.1.csv.
    #     'n_samples' : 13979592,
    #     'n_positive_t': 36711,
    #     'n_negative_t': 11845944,
    #     'n_positive_c': 4063,
    #     'n_negative_c': 2092874
    # }
    CRITEO2_RATES = uplift_calibration.CRITEO2_RATES
    # Baseline conversion rate for p_c is 0.0019375880152813366
    # Baseline conversion rate for p_t is 0.0030894610674129645
    # Start from that. Or maybe drop p_t to 0.0019 too? This will
    # satisfy the assumptions of k-undersampling and naive undersampling
    # quite well.
    pt_natural_rate = CRITEO2_RATES['n_positive_c'] /\
        (CRITEO2_RATES['n_positive_c'] + CRITEO2_RATES['n_negative_c'])
    pt_rates = [pt_natural_rate * 2**i for i in range(6)]
    # This will produce [0.0030894610674129645, 0.006178922134825929,
    # 0.012357844269651858, 0.024715688539303716, 0.04943137707860743, 0.09886275415721486]
    new_t_rate = 0.5  # Half of all observations should be treated observations, half untreated (control)
    # How do we estimate "#fraction of samples to keep"? This will be essential in getting
    # equal dataset sizes
    #c_observations_to_keep = CRITEO2_RATES['n_positive_c'] + CRITEO2_RATES['n_negative_c']
    c_observations_to_keep = CRITEO2_RATES['n_positive_t'] * 10
    # Nope. We should be keeping 10 * n_positive_t observations.
    frac_observations_to_keep = 2 * c_observations_to_keep / CRITEO2_RATES['n_samples']
    # We want to keep the maximum number of observations that satisfies the
    # set criteria, i.e. n_t=n_c, i.e. the control group will always be the
    # same size as will the treatment group.
    data_format = load_data.CRITEO_FORMAT
    data_format['file_name'] = './datasets/criteo2/criteo-research-uplift-v2.1.csv'

    for rate in pt_rates:
        uplift_calibration.get_semi_simulated_data(
            data_format, CRITEO2_RATES,
            new_t_rate=new_t_rate,
            new_t_conversion_rate=rate,
            frac_samples_to_keep=frac_observations_to_keep,
            output_filename="datasets/conversion_rate_experiments/criteo2_" + str(rate) + '.csv'  # What should the name be?
        )

# Generate slurm scripts for this.
# dataset_name(!!), undersampling, model, (correction dependent on undersampling)
# DC-LR and Uplift RF most interesting.
# Aim to show that k-undersamplin
def generate_slurm_parameters(param_file='./tmp/parameters.txt'):
    """
    Slurm-script generation over
    1. Datasets
    2. Subsampling of training and validation sets
    3. Undersampling schemes
    4. Models (also baselines for every dataset!)

    Looking for optimal model for every dataset and undersampling scheme,
    i.e. not that interested in performance over different k.
    -Uplift RF still needs to be split into multiple jobs (?) at least
     for the larger datasets, i.e. cannot write the results directly to
     _one_ file, but will have to parse later. Perhaps write to one
     directory?

    Args:
    param_file (str): File name and path of file to write parameters to.
    """
    parameter_file = open(param_file, 'w')
    # Pass k-values as 'none' if you want undersampling_experiments to handle k-selection
    # Pass as 2.0 or 2.0,3.4 for k-undersampling and split undersampling with specified k (k_t and k_c).

    for dataset in datasets:
        for model in models:
            for undersampling_scheme in undersampling_schemes:
                # Generate changing line for slurm
                # Generate file names.
                txt = './' + path
                txt += dataset + ' '
                txt += str(undersampling_scheme) + ' '
                txt += str(model) + ' '
                if undersampling_scheme == 'none':
                    correction = 'none'
                elif undersampling_scheme == 'k_undersampling':
                    correction = 'div_by_k'
                elif undersampling_scheme == 'split_undersampling':
                    correction = 'analytic'
                txt += correction + ' '
                # Lastly result file
                # Name them with all details  # Not rate in ouput file name?
                output_file_name = './results/' + dataset  # Write everything to result-directory
                output_file_name += '_' + undersampling_scheme
                output_file_name += '_' + model + '_' + correction
                output_file_name += '.csv'
                txt += output_file_name + ' '
                # Add k-value(s) as floats or str 'none'/'all' to make program go through all values.
                # k-values:
                # For dc-lr: all in one? Does the data-loader require this to be restarted? I am overwriting the
                # training and validation sets...
                # for cvt-lr: same as for dc-lr
                # for uplift-rf: all separately.
                # for uplift-nn: all separately
                # Nah. Just run all experiments separately. "Sombody else's problem."
                # Uplift-NN and CVT-LR does not support analytic correction (and hence split undersampling)
                # and can hence be included in a limited sense in the experiments. Although a
                # neural net would need gpu-allocation from ukko. Write that in a separate file.
                if undersampling_scheme == 'split_undersampling':
                    k_values = product([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                                        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
                    # Make into list
                    k_values = [*k_values]
                elif undersampling_scheme == 'none':
                    k_values = [1.0]
                else:
                    k_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
                for k in k_values:
                    if undersampling_scheme == 'split_undersampling':
                        tmp_txt = txt + str(k[0]) + ',' + str(k[1]) + ' '
                    else:
                        tmp_txt = txt + str(k) + ' '
                    # Write to file:
                    parameter_file.write(tmp_txt + '\n')
                # Have separate slurm script that fetches these as needed.
    # Close file:
    parameter_file.close()


if __name__ == '__main__':
    generate_data()
