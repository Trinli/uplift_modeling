"""
Functions and script for generating data from criteo2
with original conversion rates, but subsampled so that
there are equally many treatment and control samples.
We do this for "full" data, half of that, etc up to
1/128 of the "full" (which is around 4M samples).

"""

import models.uplift_calibration as uplift_calibration
import data.load_data as load_data

def generate_data():
    """
    Function for generating subsamples of criteo-dataset where
    the treatment:control rate is 1:1 and the total sample size
    starts with max and drops by half for additional 7 iterations.
    """
    # The maximum number of samples if we have equally many
    # treated and untreated samples is:
    samples = (uplift_calibration.CRITEO_RATES['n_positive_c'] +\
        uplift_calibration.CRITEO_RATES['n_negative_c']) * 2
    format = load_data.CRITEO_FORMAT
    format['file_name'] = 'criteo-uplift.csv'

    for i in range(8):
        # File name with "1_of_64"
        # Own folder
        output_file = 'generated_subsamples/' + str(2**i) + '.csv'
        # Fraction of total dataset:
        rate = samples / (2**i) / uplift_calibration.CRITEO_RATES['n_samples']
        if rate == 0.3082366521764452:
            # Apparently there is a difference on the 16th decimal on my Mac and on
            # the university cluster.
            rate = 0.3082366521764451
        print("Generating subsample for rate {}... ".format(rate))
        # Generate data:
        uplift_calibration.get_semi_simulated_data(format,
                                                  uplift_calibration.CRITEO_RATES,
                                                  new_t_rate=.5, frac_samples_to_keep=rate,
                                                  output_filename=output_file,
                                                  path='./datasets/criteo1/',)
        print("Done.")


if __name__ == '__main__':
    generate_data()
