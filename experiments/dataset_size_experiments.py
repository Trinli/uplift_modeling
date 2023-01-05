"""
Code for dataset size experiments.

1. Generate suitable datasets.
-Both criteo1 and criteo2 (we might drop one later)
-Sizes: 1M:1M, 750k, 500k, 250k, 150k, 100k, 75k, 50k
-Name them following some systematic pattern (that is easy to reproduce later
when reading in the data!)
"""
import os
import data.load_data
import models.uplift_calibration as uplift_calibration
from itertools import product
from random import uniform

# Sizes of treatment and control groups in datasets to generate
#sizes = [1000000, 750000, 500000, 250000, 150000, 100000, 75000, 50000]
# Define subsampling in terms of rate of "original" in the experiments.
rates = [1, .75, .5, .25, .15, .1, 0.05, 0.025, 0.0125]
datasets = ['criteo1', 'criteo2']
# To test the code, untoggle:
#sizes = [5000, 1000]
#datasets = ['test']

undersampling_schemes = ['none', 'k_undersampling', 'split_undersampling']
models = ['dc_lr', 'uplift_rf']  # Run separately with 'cvt-lr' (drop split undersampling)
# and 'uplift-nn' (again, drop split undersampling. Call it with a different slurm script
# allocating gpu's.)
# One graph will cover one dataset, one model, and undersampling scheme over _all_ sizes.
# -Store them like this for easy access?


def split_data(file_name, training_fraction=0.5):
    """
    Function for splitting csv-file into two randomly with
    50/50 approximate split.
    -Not used in experiments.

    Args:
    file_name (str): Name of input file to be split into two
    training_fraction (float): In [0, 1], the fraction of observations
     to include in the training set. The rest is written to the
     testing set file.
    """
    # Open a 'training_set-file' and a 'testing_set-file' to write to
    training_file = open(file_name + 'training_set' + '.csv', 'w')
    testing_file = open(file_name + 'testing_set' + '.csv', 'w')
    # Open csv-file to be read
    with open(file_name, 'r') as handle:
        for line in handle:
            # Randomize
            if uniform(0, 1) < training_fraction:
                # Write into training file.
                training_file.write(line)
            else:
                # Write into testing file
                testing_file.write(line)


def generate_data():
    """
    Not used in experiments.
    """
    for dataset in datasets:
        if dataset == 'test':
            file_name = 'criteo_100k.csv'
            data_format = data.load_data.CRITEO_FORMAT
            data_rates = uplift_calibration.TEST_RATES
        elif dataset == 'criteo1':
            file_name = 'criteo1/criteo-uplift.csv'
            data_format = data.load_data.CRITEO_FORMAT
            data_rates = uplift_calibration.CRITEO_RATES
        elif dataset == 'criteo2':
            file_name = 'criteo2/criteo-research-uplift-v2.1.csv'
            data_format = data.load_data.CRITEO_FORMAT
            data_rates = uplift_calibration.CRITEO2_RATES
        else:
            print("Dataset must have value 'criteo1', 'crtieo2', or 'test'")
            raise ValueError
        data_format['file_name'] = file_name

        for size in sizes:
            # Generate file name for new file:
            # Perhaps also output path to keep 
            output_file = file_name + str(size) + '.csv'
            # Estimate fraction of samples to keep:
            # data_format['n_samples'] is the total number of samples.
            # We want to keep 'size' number of treated and 'size' number of control observations
            # i.e. 2*size:
            frac = 2 * size / data_rates['n_samples']
            # Check if path exists
            path_ = './datasets/size_experiments/'
            if not os.path.isdir(path_): # Add criteo*/ to path here?
                # Create dir
                os.mkdir(path_)
            if not os.path.isdir(path_ + 'criteo1/'):
                os.mkdir(path_ + 'criteo1/')
            if not os.path.isdir(path_ + 'criteo2/'):
                os.mkdir(path_ + 'criteo2/')
            # Create dataset (stores to file)
            uplift_calibration.get_semi_simulated_data(
                data_format, data_rates, new_t_rate=0.5,
                frac_samples_to_keep=frac,
                path='./datasets/',
                output_filename= 'size_experiments/' + output_file
            )


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
    for rate in rates:
        for dataset in datasets:
            for model in models:
                for undersampling_scheme in undersampling_schemes:
                    # Generate changing line for slurm
                    # Generate file names.
                    txt = ''
                    # Filename done. Next parameters
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
                    # Name them with all details
                    output_file_name = './results/' + dataset  # Write everything to result-directory
                    output_file_name += '_' + undersampling_scheme
                    output_file_name += '_' + model + '_' + correction
                    output_file_name += '_r' + str(rate)
                    output_file_name += '.csv' + ' '
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
                            tmp_txt = txt + str(k[0]) + ',' + str(k[1])+ ' '
                        else:
                            tmp_txt = txt + str(k) + ' '
                        # Add downsampling rate:
                        tmp_txt += str(rate)
                        # Write to file:
                        parameter_file.write(tmp_txt + '\n')
                    # Have separate slurm script that fetches these as needed.
    # Close file:
    parameter_file.close()


if __name__ == '__main__':
    #generate_data()
    generate_slurm_parameters()
