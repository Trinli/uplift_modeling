"""
Here are _two_ implementations of isotonic regression for uplift modeling.

UpliftIsotonicRegression uses Athey & Imbens' revert-label aproach (2016?).
Use this. This approach does not guarantee equally many treatment and
control samples in one bin, but the PAVA will lead to a result where the
bins approximately contain equally many treatment and control samples by
simply merging bins until the predicted probabilities map to scores
monotonically.
-This is theoretically justified.

UpliftIR is work-in-progress. The big difference is that this version tries
to estimate bin probabilities by weighting samples so that each bin would
hypothetically contain equally many treatment and control samples. The previous
version does none of that.
"""

import warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d


class UpliftIsotonicRegression(object):
    """
    Variant of isotonic regression-based calibration for
    uplift modeling using Athey & Imbens' revert-label.
    """
    def __init__(self, y_min=-.999, y_max=.999):
        """
        Args:
        y_min (float): Smallest uplift this model can predict
        y_max (float): Largest uplift this model can predict
        """
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max,
                                        out_of_bounds='clip')

    def fit(self, x_vec, y_vec, t_vec):
        """
        Args:
        x_vec (np.array): 1d array of scores
        y_vec (np.array): Label, True indicates positive
        t_vec (np.array): group, True indicates treatment group
        """
        # 1. Define 'r' as the revert label
        p_t = sum(t_vec) / len(t_vec)
        # This does not seem right:
        r_vec = np.array([y_i * (t_i - p_t) / (p_t * (1 - p_t)) for\
                          y_i, t_i in zip(y_vec, t_vec)])

        # 2. Run isotonic regression on the new problem
        self.model.fit(x_vec, r_vec)

    def predict_uplift(self, x_vec):
        """
        Method for predicting calibrated change in probability.

        Args:
        x_vec (np.array): Scores to convert to probabilities
        """
        return self.model.predict(x_vec)


class UpliftIR(object):
    """
    Exotic version of isotonic regression for uplift modeling.
    Needs better theory.
    Predictions fall in [-1, 1].
    """
    def __init__(self, min_prob=-.999, max_prob=.999):
        """
        Args:
        min_prob (float): Smallest uplift probability
        max_prob (float): Largest uplift probability
        """
        warnings.warn("UpliftIR is work-in-progress. Use " +\
                      "UpliftIsotonicRegression instead!")
        assert -1 <= min_prob <= 1, "min_prob needs to be in [-1, 1]"
        assert -1 <= max_prob <= 1, "max_prob needs to be in [-1, 1]"
        assert min_prob < max_prob, "max_prob needs to be larger than min_prob"
        self.min_prob = min_prob
        self.max_prob = max_prob
        # Initialize model at fitting stage
        self.model = None

    def fit(self, x, y, t, verbose=False):
        """
        Method for fitting isotonic regression model for uplift modeling.

        Args:
        x (np.array): score
        y (np.array): label ('True' for positive)
        t (np.array): Group label ('True' for treatment)
        """
        # Sort samples in increasing order:
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        t = t[idx]

        # Transform into bins
        bins = []  # List retains order
        for i, group in enumerate(t):
            if group:
                # Create new bin of _one_ treatment sample:
                bins.append(Bin(x[i], x[i], y[i], 1, 0, 0))
            elif not group:
                bins.append(Bin(x[i], x[i], 0, 0, y[i], 1))
        # Merge samples with equal scores into bins
        # Samples with equal scores _have_ to fall into the
        # same bin per definition!
        # Compare *this bin to the next, hence iterate until [:-1].
        i = 0
        while True:
            if i == len(bins) - 1:
                break
            if bins[i].score_min == bins[i + 1].score_min:
                # Does the 'bin'-object get updated in the bins list?
                bins[i].merge_bins(bins[i + 1])
                # Drop bins[i + 1]
                del bins[i + 1]
            i += 1

        # Perform PAVA
        # Note that there needs to be both treatment and control
        # samples in all bins and the group samples in a bin should
        # be weighted so that they count as p(t=1)=p(t=0).
        # If bin_prob[i] > bin_prob[i + 1]
        # then merge.
        # After merge, reduce i by one, and check whether *this
        # bin should be merged to previous. After merge, reduce
        # i by one etc.
        # If no PAV-violation, move to next (i + 1).
        # End when no violation between [-2] and [-1]
        i = 0
        while True:
            if i < 0:
                i = 0
            elif i == (len(bins) - 1):
                # If we have reached the last bin:
                break
            if bins[i].prob > bins[i + 1].prob:
                # Violation found. Merge, drop and backtrack.
                bins[i].merge_bins(bins[i + 1])
                del bins[i + 1]
                i -= 1
            else:
                # Move on to next bin
                i += 1

        # Transform model into interpolation model and store
        # in self.model.
        # We have the min and max scores in bins as well as the probabilities
        # these bins should map to.
        bin_scores = []
        bin_probabilities = []
        for bin in bins:
            bin_scores.append(bin.score_min)
            bin_scores.append(bin.score_max)
            # Append the same probability twice to make a piecewise
            # constant function:
            bin_probabilities.append(bin.prob)
            bin_probabilities.append(bin.prob)
        fill_value = (max(self.min_prob, bins[0].prob),\
                      min(self.max_prob, bins[-1].prob))
        self.model = interp1d(bin_scores, bin_probabilities,
                              bounds_error=False,
                              fill_value=fill_value,
                              assume_sorted=True)

        if verbose:
            print("Number of bins: {}".format(len(bins)))
            print("Smallest bin:")
            print("score_min: {}, prob: {}".format(bins[0].score_min,
                                                   bins[0].prob))
            print("Probability estimated from only one group: {}".format(
                bins[0].prob_estimated_only_from_one_group))
            print("Largest bin:")
            print("score_max: {}, prob: {}".format(bins[-1].score_max,
                                                   bins[-1].prob))

    def predict_uplift(self, x):
        """
        Method for predicting uplift

        Args:
        x (np.array): 1 * n size array of scores to be converted to
         uplift probabilities using the ir-model.
        """
        return self.model(x)


class Bin(object):
    """
    Auxiliary class for isotonic regression for uplift modeling.
    """
    def __init__(self, score_min, score_max, y_t, n_t, y_c, n_c):
        """
        Args:
        y_t (int): Number of positive treatment samples
        n_t (int): Number of treatment samples
        ...
        N_t_tot (int): Total number of treatment samples in data
        N_c_tot (int): Total number of control samples in data

        Notes:
        One treatment samples should act as n_c / (n_t + n_c) samples
        for probability estimates, but this is better enforced in
        the probability estimate method.
        """
        # One treatment sample should act as (.) control samples
        # for weighted averages, although this could also be
        # enforced on bin-by-bin basis. That would make a lot
        # more sense.
        self.score_min = score_min
        self.score_max = score_max
        self.num_y_t = y_t
        self.num_y_c = y_c
        self.num_n_t = n_t
        self.num_n_c = n_c
        # Method sets self.prob
        self.bin_probability()

    def merge_bins(self, bin_2):
        """
        Method for merging two neighboring bins. bin_2
        has to be deleted separately.
        """
        self.num_y_t += bin_2.num_y_t
        self.num_n_t += bin_2.num_n_t
        self.num_y_c += bin_2.num_y_t
        self.num_n_c += bin_2.num_n_c
        self.score_min = min(self.score_min, bin_2.score_min)
        self.score_max = max(self.score_max, bin_2.score_max)
        self.bin_probability()

    def bin_probability(self):
        """
        Method for estimating probability for a bin.
        The factors 2 * and (-2) * actually corresponds to the
        revert label approach proposed by Athey and Imbens (2016?)
        but here on a bin-by-bin basis.
        Estimating bin probabilities from only treatment or control
        samples is also strange, but it needs to be like
        this for correctness of the PAVA. The PAVA will sort them out
        anyway save for samples in the first bin, the last bin, and
        possibly in some middle bin mapping to p=0.
        """
        if self.num_n_t == 0:
            self.prob = (-2) * self.num_y_c / self.num_n_c
            self.prob_estimated_only_from_one_group = True
        elif self.num_n_c == 0:
            self.prob = 2 * self.num_y_t / self.num_n_t
            self.prob_estimated_only_from_one_group = True
        else:
            # Estimating probabilities as frequencies with no prior
            # mean() handles the weighting of samples!
            self.prob = np.mean([
                2 * self.num_y_t / self.num_n_t,
                (-2) * self.num_y_c / self.num_n_c])
            self.prob_estimated_only_from_one_group = False
