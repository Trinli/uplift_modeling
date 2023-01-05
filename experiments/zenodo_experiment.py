"""
Experiments with undersampling on zenodo-data.

1. The data needs to be changed to have a conversion rate of approx 0.5%.
2. Run experiments...
"""

import models.uplift_calibration as uea # Contains function for changing positive rates etc.
import data.load_data as ld


def generate_data():
    file_name = 'datasets/Zenodo/uplift_synthetic_data_100trials.csv'
    out_file = './datasets/zenodo_modified.csv'
    # First, we need the data rates
    # What conversion rates do we want? Something around 0.5%
    # Drop positive observations in t and c with equal probability.
    t_rate = 0.002  # Positive rate for treated observations
    # Probability of keeping a positive observation:
    # FIX FORMULA!
    # 0.005 = p_keep * n_pos / (p_keep * n_pos + n_neg)
    # p_keep = 0.005  * (p_keep * n_pos + n_neg) / n_pos
    # p_keep - 0.005 p_keep * n_pos / n_pos = 0.005 * n_neg / n_pos
    # p_keep (1 - 0.005) = 0.005 * n_neg / n_pos
    # p_keep = 0.005 / (1-0.005) * n_neg / n_pos
    p_keep = 0.005 / (1- 0.005) * uea.ZENODO_RATES['n_negative_t'] / uea.ZENODO_RATES['n_positive_t']
    n_pos_c = int(p_keep * uea.ZENODO_RATES['n_positive_c'])
    c_rate = n_pos_c / (n_pos_c + uea.ZENODO_RATES['n_negative_c'])
    ld.ZENODO_FORMAT['file_name'] = file_name
    uea.get_semi_simulated_data(ld.ZENODO_FORMAT, uea.ZENODO_RATES,
        new_t_conversion_rate=t_rate, new_c_conversion_rate=c_rate,
        output_filename=out_file)
