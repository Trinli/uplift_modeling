"""
Undersampling experiments (extended).
This file contain
-functions for undoing effect of undersampling, including div by k,
 and analytic correction (important!). A calibration function for
 uplift is implemented in isotonic_regression.py.
-code for downsampling data

Misc

# If we set conversion rate for VOTER dataset in treatment group to 0.005 and
# drop positive control group observations with the same probability, then
# the correct conversion rate for the control group is 0.003475:
>>>  a = (0.005 * VOTER_RATES['n_negative_t']) / ((1 - 0.005) * VOTER_RATES['n_positive_t'])
>>> a
0.008270678785254712
>>> r2 = a * VOTER_RATES['n_positive_c'] / ( a * VOTER_RATES['n_positive_c'] + VOTER_RATES['n_negative_c'])
>>> r2
0.0034759814461150563
"""
from uplift_modeling.data.load_data import DATA_FORMAT
import numpy as np
import csv
import random


CRITEO_RATES = {
    # This is so that we can estimate conversion rates
    # up front and do probabilistic sampling one row
    # at a time. This removes the need for almost any
    # working memory.
    'n_samples': 25309482,
    'n_positive_t': 51258,
    'n_negative_t': 21357569,
    'n_positive_c': 6803,
    'n_negative_c': 3893852
}

CRITEO2_RATES = {
    # Criteo published a second dataset called
    # criteo-research-uplift-v2.1.csv.
    'n_samples' : 13979592,
    'n_positive_t': 36711,
    'n_negative_t': 11845944,
    'n_positive_c': 4063,
    'n_negative_c': 2092874
}

HILLSTROM_RATES = {
    # Rates of types of samples in the Hillstrom-dataset.
    'n_samples': 42613,
    'n_positive_t': 3894,
    'n_negative_t': 17413,
    'n_positive_c': 2262,
    'n_negative_c': 19044
}

VOTER_RATES = {
    # ToDo.
    'n_samples': 229444,
    'n_positive_t': 14438,
    'n_negative_t': 23763,
    'n_positive_c': 56730,
    'n_negative_c': 134513
}

ZENODO_RATES = {
    'n_samples': 1000000,
    'n_positive_t': 179487,
    'n_negative_t': 320513,
    'n_positive_c': 124264,
    'n_negative_c': 375736
}


TEST_RATES = {
    # These are the number of samples in the Criteo100k.csv-file.
    'n_samples': 100000,
    'n_positive_t': 198,
    'n_negative_t': 84475,
    'n_positive_c': 20,
    'n_negative_c': 15307
}


def analytic_correction(predictions, k=None, p_bar=None, s=None):
    """
    Function for fixing bias caused by undersampling. We need both
    p(y|x,t=1) and p(y|x, t=0) to estimate this, although this function
    only corrects one probability at a time.

    Args:
    predictions (np.array): Predicted probabilities (e.g. p(y|x,t=1)).
    k (float): k used for undersampling. 
    p_bar (float): In [0, 1]. Average conversion rate in data. 
    s (float): Optinoal correction factor as defined below.
    """
    # Redefine 'k' by correction formula:
    if s is None:
        s = (1 / k - p_bar) / (1 - p_bar)

    def correct_one(item, s):
        return -s * item / (item - item * s - 1)

    tmp = np.array([correct_one(item, s) for item in predictions])
    return tmp


def div_by_k(predictions, k):
    """
    Is this implemented already in previous experiments?
    Pretty trivial..
    """
    return np.array([item / k for item in predictions])


def get_semi_simulated_data(data_format,
                            data_rates=CRITEO_RATES,
                            new_t_rate=None,
                            new_t_conversion_rate=None,
                            new_c_conversion_rate=None,
                            frac_samples_to_keep=None,
                            path='./',
                            output_filename='tmp.csv',
                            verbose=True):
    """ 
    Fumction for creating semi-simulated dataset as csv file
    with defined treatment rates, positive rate of both treatment
    and control samples separately, and fraction of original dataset
    to keep. The functions in load_data can then handle this csv-file.

    -If n_samples or frac is not defined, the function will keep
     as many samples as possible.
    -If t_conversion_rate or c_conversion_rate is not defined, the
     function will keep the conversion rates that are in the
     data.
    -If t_rate is not defined, the function will keep the treatment
     rate that is in the data.
    -We do probabilistic sampling, i.e. we read the input file line
     by line and decide whether to include that line in the output
     file. For this, we need to know the number of different types
     of samples in the data up-front (to not have to scan the data).
     This info is already stored above in e.g. CRITEO_RATES.

    Args:
    data_format ({}): Format as defined in load_data.py including
     file name, class label idx, treatment idx, treatment labels.
    data_rates (dict): Dict with info on positive and negative number
     of samples in data split by treated/untreated. This way we do
     not need to scan the entire data before estimating probabilities
     for keeping samples.
    new_t_rate (float): Desired rate of treated samples of all
     in semi-simulated data.
    new_t_conversion_rate (float): In [0, 1]. Desired conversion
     rate in the semi-simulated data for treated samples.
    new_c_conversion_rate (float): In [0, 1]. Desired conversion
     rate in the semi-simulated data for control samples.
    n_samples (int): If not none, the sample will be reduced to
     this size.
    path (str): Location of file unless in working directory.
    output_filename (str): Name of file to create and write
     undersampled data to.

    Misc:
    -If frac is too large, should function return an error or create
     as large as possible dataset that will have smaller N than requested?
     Latter. Otherwise I might end up training a model that does not
     have the input size as I think it should. Let output be informative,
     though, like "with these rates, a frac of x is max."
    """
    # 1. Check which variables are defined. Define undefined.
    tmp_t_n = data_rates['n_positive_t'] + data_rates['n_negative_t']
    tmp_c_n = data_rates['n_positive_c'] + data_rates['n_negative_c']
    old_t_rate = tmp_t_n / (tmp_t_n + tmp_c_n)
    if new_t_rate is None:
        # Set to treatment rate present in data:
        new_t_rate = old_t_rate
    old_t_conversion_rate = data_rates['n_positive_t'] /\
        (data_rates['n_positive_t'] + data_rates['n_negative_t'])
    if new_t_conversion_rate is None:
        # Set to conversion rate in data
        new_t_conversion_rate = old_t_conversion_rate
    old_c_conversion_rate = data_rates['n_positive_c'] /\
        (data_rates['n_positive_c'] + data_rates['n_negative_c'])
    if new_c_conversion_rate is None:
        # Set to conversion rate in data
        new_c_conversion_rate = old_c_conversion_rate

    # 2. Estimate number of treatment-samples _without_ frac:
    if new_t_conversion_rate >= old_t_conversion_rate:
        # Drop negative:
        new_n_positive_t = data_rates['n_positive_t']
        # The next one is not a float now. Fix at some point?
        new_n_negative_t = data_rates['n_positive_t'] *\
            (1 - new_t_conversion_rate) / new_t_conversion_rate
    elif new_t_conversion_rate < old_t_conversion_rate:
        # Drop positive:
        new_n_negative_t = data_rates['n_negative_t']
        new_n_positive_t = data_rates['n_negative_t'] *\
            new_t_conversion_rate / (1 - new_t_conversion_rate)
    if new_c_conversion_rate >= old_c_conversion_rate:
        # Drop negative
        new_n_positive_c = data_rates['n_positive_c']
        # print(data_rates['n_negative_c'])
        new_n_negative_c = data_rates['n_positive_c'] *\
            (1 - new_c_conversion_rate) / new_c_conversion_rate
        # print(new_n_negative_c)
    elif new_c_conversion_rate < old_c_conversion_rate:
        new_n_negative_c = data_rates['n_negative_c']
        new_n_positive_c = data_rates['n_negative_c'] *\
            new_c_conversion_rate / (1 - new_c_conversion_rate)

    # 3. Next reduce the number of samples so that the ratio
    # between treated and untreated is as defined by parameter.
    new_n_samples_t = new_n_positive_t + new_n_negative_t
    new_n_samples_c = new_n_positive_c + new_n_negative_c
    # tmp_t_rate, i.e. current treatment rate if we sampled the
    # number of samples estimated above:
    tmp_t_rate = new_n_samples_t /\
        (new_n_samples_t + new_n_samples_c)
    if tmp_t_rate <= new_t_rate:
        # Treatment rate with numbers above exceeds desired.
        # Drop control samples:
        tmp_frac = new_n_samples_t * (1 - new_t_rate) / new_t_rate / new_n_samples_c
        new_n_positive_c *= tmp_frac
        new_n_negative_c *= tmp_frac
        new_n_samples_c = new_n_positive_c + new_n_negative_c
    elif tmp_t_rate > new_t_rate:
        # Drop treatment samples.
        tmp_frac = new_n_samples_c * new_t_rate / (1 - new_t_rate) / new_n_samples_t
        new_n_positive_t *= tmp_frac
        new_n_negative_t *= tmp_frac
        new_n_samples_t = new_n_positive_t + new_n_negative_t

    # 4. Check if we can make a sample of size N * frac_samples_to_keep
    # of this.
    # ~sum of samples exceeds desired? Drop samples from both groups equally
    # sum is less? Cannot create dataset with defined aprameters. Print
    # maximal frac for these rates.
    # ???:
    tot_tmp_samples = new_n_samples_t + new_n_samples_c
    if frac_samples_to_keep is not None:
        max_frac = tot_tmp_samples / data_rates['n_samples']
        tmp_txt = "Cannot form dataset with given parameters. \n" +\
            "With given rates, a fraction of maximally " +\
            "{} can be produced.".format(max_frac)
        assert max_frac >= frac_samples_to_keep, tmp_txt
        # At this point, max_frac should be possible to produce.
        # Multiplier to change new rates into what will ultimately
        # be produced:
        tmp = (frac_samples_to_keep * data_rates['n_samples']) /\
            tot_tmp_samples
        new_n_positive_t *= tmp
        new_n_negative_t *= tmp
        new_n_positive_c *= tmp
        new_n_negative_c *= tmp

    # 5. Divide all by number of respective original samples to get
    # sampling probabilities for all types of samples separately!
    # Print them for now to test the code. Then start with the file reading.
    # Sampling rates:
    sr_pos_t = new_n_positive_t / data_rates['n_positive_t']
    sr_neg_t = new_n_negative_t / data_rates['n_negative_t']
    sr_pos_c = new_n_positive_c / data_rates['n_positive_c']
    sr_neg_c = new_n_negative_c / data_rates['n_negative_c']
    if verbose:
        print("Pos_t: {}, Neg_t: {}".format(sr_pos_t, sr_neg_t))
        print("Pos_c: {}, Neg_c: {}".format(sr_pos_c, sr_neg_c))
    # Might have to cap all values at 1.0 to not have issues with numeric
    # stability. Seems to be correct.

    # 6. Open files both for reading and writing. Analyze samples
    # one at a time and write to file if sampled. This should save
    # working memory.

    def parse_row(row, data_format, output_file_handle):
        # Let function write to parent function variables.
        if row[data_format['t_idx']] in data_format['t_labels']:
            # Add row somewhere only if treatment label matches the ones
            # defined in format
            if row[data_format['t_idx']] == data_format['t_labels'][0]:
                # Idx '0' are the treated samples.
                if 'y_labels' in data_format.keys():
                    # I.e. if key y_label exists
                    if row[data_format['y_idx']] == data_format['y_labels'][0]:
                        # Positive label is first:
                        # Include sample with t_pos_sampling_prob
                        if random.random() < sr_pos_t:
                            output_file_handle.writerow(row)
                        # Otherwise skip row.
                    elif row[data_format['y_idx']] == data_format['y_labels'][1]:
                        if random.random() < sr_neg_t:
                            output_file_handle.writerow(row)
                    else:
                        print("Y-label in data ('{}') not in data_format.".format(row[data_format['y_idx']]))
                        raise Exception("Unknown y-label encountered")
                else:
                    # If y_labels not defined in data_format
                    if row[data_format['y_idx']] == str(1):
                        if random.random() < sr_pos_t:
                            output_file_handle.writerow(row)
                    elif row[data_format['y_idx']] == str(0):
                        if random.random() < sr_neg_t:
                            output_file_handle.writerow(row)
                    else:
                        print("Y-label in data ('{}') not defined in data_format.".format(row[data_format['y_idx']]))
                        raise Exception("Unknown y-label encountered")
            elif row[data_format['t_idx']] == data_format['t_labels'][1]:
                # Control sample!
                if 'y_labels' in data_format.keys():
                    # I.e. if key y_label exists
                    if row[data_format['y_idx']] == data_format['y_labels'][0]:
                        # Positive label is first:
                        if random.random() < sr_pos_c:
                            output_file_handle.writerow(row)
                    elif row[data_format['y_idx']] == data_format['y_labels'][1]:
                        if random.random() < sr_neg_c:
                            output_file_handle.writerow(row)
                    else:
                        print("Y-label in data ('{}') not in data_format.".format(row[data_format['y_idx']]))
                        raise Exception("Unknown y-label encountered")
                else:
                    # If y_labels not defined in data_format
                    if row[data_format['y_idx']] == str(1):
                        if random.random() < sr_pos_c:
                            output_file_handle.writerow(row)
                    elif row[data_format['y_idx']] == str(0):
                        if random.random() < sr_neg_c:
                            output_file_handle.writerow(row)
                    else:
                        print("Y-label in data ('{}') not defined in data_format.".format(row[data_format['y_idx']]))
                        raise Exception("Unknown y-label encountered")
            else:
                # It should be impossible to reach this.
                print("T-label found in data ('{}') not in data_format".format(row[data_format['t_idx']]))
                raise Exception("Unknown t-label encountered.")
        return

    # Next read and parse the actual file:
    output_handle = open(path + output_filename, 'w')
    output_file = csv.writer(output_handle)
    # Here we could also open a separate file for writing and
    # store the results line by line.
    # new_file = csv.writer(output_handle)
    print("Filename and path: {}, {}".format(output_filename, path))
    with open(path + data_format['file_name'], "r") as handle:
        file_ = csv.reader(handle)
        if data_format['headers']:  # If there are headers,
            # store and move to next.
            headers = next(file_)
            output_file.writerow(headers)
        for row in file_:
            # parse_row() is called here rather than writing all
            # rows to memory first and then dealing with that
            # as the alternative would use 2x the amount of
            # working memory. This is essential for the Criteo
            # dataset.
            # tmp_data.append(row)
            parse_row(row, data_format, output_file)
    output_handle.close()
    return


def test_mini_criteo():
    """
    Code using local criteo100k.csv file to test the above.
    """
    import data.load_data as load_data
    data_format = load_data.DATA_FORMAT
    # Use different file:
    data_format['file'] = 'criteo100k.csv'
    get_semi_simulated_data(data_format, TEST_RATES,
                            .50, .001, .0008, 10000)


def test_new_criteo():
    """
    Does the code work with the newer criteo dataset (criteo-uplift2.csv)?
    """
    import data.load_data as load_data
    data_format = load_data.DATA_FORMAT
    data_format['file'] = "criteo2/criteo-research-uplift-v2.1.csv"
    get_semi_simulated_data(data_format, data_rates=CRITEO2_RATES, new_t_rate=.5)
