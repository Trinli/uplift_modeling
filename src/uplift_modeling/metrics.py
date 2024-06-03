"""
Functions for estimating the most common uplift metrics and some Bayesian
statistics based on the beta-distribution and the beta-difference distribution.
The UpliftMetrics-class estimates most metrics with one function call efficiently.
This class also contains methods for writing the results to a csv-file.
"""

import csv
from datetime import datetime, timezone
import warnings
from numba import jit
import numpy as np
from pathlib import Path
from scipy.stats import beta


class UpliftMetrics():
    """
    Class for a collection of metrics relating to uplift modeling.
    The main purpose of this class is to keep track of metrics
    and document the results together with appropriate test
    descriptions. Initialization causes estimation of all metrics
    and the save_to_csv()-method stores the metrics.

    Parameters
    ----------
    data_class : np.array([bool])
        Array of class labels for observations.
    data_prob : np.array([float]) 
        Array of uplift predictions as probabilities for observations. 
        Can also be replaced with data_score, but metrics 
        relying on probabilitise will be way off.
    data_group : np.array([bool])
        Array of group labels for observations. True indicates that 
        observation belongs to the treatment group.
    test_name : str
        Will be written to the result file when write_to_csv() is called.
    test_description : str
        Will be written to the result file when write_to_csv() is called.
    algorithm : str
        type of algorithm, e.g. 'double-classifier with lr'
    dataset : str
        dataset used to obtain model and metrics
    parameters : str?
        Will be stored.
    estimate_qini : bool
        Decided whether the qini-coefficient is estimated.
    n_bins : int
        Number of bins used for estimation of expected uplift calibration
        error and Kendall's tau. Sometimes this is referred to as 'k'.
    """
    def __init__(self, data_class, data_prob, data_group,
                 test_name=None, test_description=None,
                 algorithm=None,
                 dataset=None,
                 parameters=None,
                 estimate_qini=False,
                 n_bins=100):

        self.algorithm = algorithm
        self.dataset = dataset
        self.parameters = parameters
        self.test_name = test_name
        self.test_description = test_description

        # Sanity check:
        if len(data_class) != len(data_prob) != len(data_group):
            raise Exception("Class, probability, and group vectors " +
                            "needs to be of equal length!")

        # Sort dataset once (largest uplift first)
        data_idx = np.argsort(data_prob)[::-1]
        data_class = data_class[data_idx]
        data_prob = data_prob[data_idx]
        data_group = data_group[data_idx]

        # Calculate conversion-rate list
        tmp_1 = _expected_conversion_rates(data_class,
                                           data_prob,
                                           data_group)
        # Estimate metrics based on this:
        self.e_r_conversion_rate = np.mean(tmp_1)
        self.conversion_rate_no_treatments = tmp_1[0]
        self.conversion_rate_all_treatments = tmp_1[-1]
        self.e_conversion_rate_random = np.mean([self.conversion_rate_no_treatments,
                                                 self.conversion_rate_all_treatments])
        self.auuc = self.e_r_conversion_rate -\
            self.e_conversion_rate_random * 1
        # Relative improvement to random.
        if self.e_conversion_rate_random != 0:
            self.improvement_to_random = self.auuc /\
                self.e_conversion_rate_random
        if estimate_qini:
            self.qini_coefficient = qini_coefficient(
                data_class, data_prob, data_group)
        else:
            self.qini_coefficient = None
        self.n_bins = n_bins  # What's a good default value?
        tmp_2 = expected_uplift_calibration_error(data_class, data_prob, data_group,
                                                  k=self.n_bins)
        self.euce = tmp_2[0]  # Extract EUCE from tuple
        self.muce = tmp_2[1]  # Extract MUCE from tuple
        # self.kendalls_tau = kendalls_uplift_tau(data_class, data_prob, data_group,
        #                                         k=self.n_bins) 
        self.kendalls_tau = None  # There seems to be some bug in this metric.
        self.unique_scores = len(np.unique(data_prob))
        self.observations = len(data_prob)
        self.adjusted_e_mse = estimate_adjusted_e_mse(data_class, data_prob, data_group)

    def __str__(self):
        """
        Method for e.g. easy printing of metrics to screen.
        """
        txt = "-" * 40 + "\n" +\
              "Test name: {0.test_name}\n".format(self) +\
              "Algorithm: {0.algorithm}\n".format(self) +\
              "dataset: {0.dataset} \n".format(self) +\
              "Test description: {0.test_description}\n".format(self) +\
              "E_r(conversion rate) \tAUUC \t\tEUCE \t\tMUCE \t\t" +\
              "Improvement to random  \tAdjusted E(MSE) \n" +\
              "{0.e_r_conversion_rate:9.9} \t\t{0.auuc:9.9} \t".format(self) +\
              "{0.euce:9.9} \t{0.muce:9.9} \t".format(self) +\
              "{0.improvement_to_random:9.9} \t\t".format(self) +\
              "{0.adjusted_e_mse:9.9} \n".format(self) +\
              "-" * 40
        # \tQini-coefficient  Kendall's tau \t
        #"{0.qini_coefficient} \t\t\t".format(self) +\
        #"{0.kendalls_tau:9.9} \t".format(self) +\  # There is something wrong with this.
        return txt

    def write_to_csv(self, file_='uplift_results.csv'):
        """
        Method for storing metrics to csv-file in predefined format.
        The function will by default append the results to the end of the file
        unless the file does not exist in which case it creates that file first.

        Parameters
        ----------
        file_ : str
            Filename for the csv-file to write results to. The current working
            directory is used if no path is included.
        """
        # 1. Check if file exists. If it does, append results. Otherwise
        # create first and add header row!
        write_new_headers = True
        if Path('./' + file_).exists():
            write_new_headers = False

        with open(file_, 'a') as resultfile:
            result_writer = csv.writer(
                resultfile, delimiter=';', quotechar='"')
            if write_new_headers:
                # Write new headers to file
                headers = ['Test name', 'Dataset', 'Test description',
                           'Algorithm', 'Parameters',
                           'Timestamp', 'E_r(conversion rate)',
                           'AUUC', 'Improvement to random [%]', 'Qini-coefficient',
                           'EUCE', 'MUCE', 'k',
                           '#Unique scores', '#Observations',
                           'E(converison rate|No treatments)',
                           'E(conversion rate|All treatments)',
                           'E(conversion rate|random)',
                           'Kendalls tau', 'Adjusted E(MSE)']
                try:
                    result_writer.writerow(headers)
                except csv.Error:
                    print("Error in saving headers")

            # Include *everything* in result list:
            result_list = [self.test_name, self.dataset, self.test_description,
                           self.algorithm, self.parameters, str(
                               datetime.now(timezone.utc)),
                               #datetime.now(datetime.UTC)),
                           self.e_r_conversion_rate, self.auuc,
                           self.improvement_to_random, self.qini_coefficient,
                           self.euce, self.muce, self.n_bins, 
                           self.unique_scores, self.observations,
                           self.conversion_rate_no_treatments,
                           self.conversion_rate_all_treatments,
                           self.e_conversion_rate_random,
                           self.kendalls_tau,
                           self.adjusted_e_mse]
            try:
                result_writer.writerow(result_list)
            except csv.Error:
                print("Error in saving results to CSV.")
                print(result_list)


@jit(nopython=True)
def expected_conversion_rate(data_class,
                             data_score,
                             data_group,
                             k,
                             smoothing=0):
    """
    Function for estimating expected conversion rate if we
    treated k/N fraction of all observations.

    Parameters
    ----------
    data_class : numpy.array([bool]) 
        An array of labels for all observations
    data_score : numpy.array([float])
        An array of scores for every observation
    data_group : numpy.array([bool])
        An array of labels for all observations. True indicates that the 
        corresponding observation belongs to the treatment group, false 
        indicates that it belongs to the control group.
    k : int
        Number of observations that should be treated according to the
        treatment plan. The highest scoring k-observations are then treated.
    smoothing : float
        Setting smoothing to something else than 0 enables smoothing when 
        estimating the conversion rate. E.g. setting it to one corresponds 
        to Laplace-smoothing.

    Implementatdion details:
    If k/N splits a clump of equally scoring observations, they are all
    treated as the "average" of this clump, i.e. the resulting conversion
    rate is an actual expected value.

    This function uses smoothing = 0 per default. This results in estimating
    the conversion rate of zero observations to 0. This happens frequently when
    we set k to something small or something very close to N (the total
    number of observations). This could become a problem if also N is small.

    Future ideas:
    An option to smoothing could be to use a bayesian prior and
    perhaps estimate the expected value instead of maximum a posteriori
    or maximum likelihood.
    """

    if k == 0:
        # handle case where there are no observations in treatment group
        # i.e. where the conversion rate is estimated only from one group.
        # data_group == True, i.e. it is a treatment observation.
        control_conversions = np.sum(data_class[~data_group])
        control_observations = np.sum(~data_group)
        # There are no treated observations if k=0:
        conversion_rate = (control_conversions + smoothing) /\
                          (control_observations + 2 * smoothing)
    elif k == len(data_class):
        # handle case where there are no observations in control group
        treatment_conversions = np.sum(data_class[data_group])
        treatment_observations = np.sum(data_group)
        # All observations are treated:
        conversion_rate = (treatment_conversions + smoothing) /\
                          (treatment_observations + 2 * smoothing)
    else:
        # This is the "ordinary" flow.
        # Sort observations by decreasing score, i.e. the ones that should be treated first
        # are first:
        data_idx = np.argsort(data_score)[::-1]
        data_class = data_class[data_idx]
        data_score = data_score[data_idx]
        data_group = data_group[data_idx]

        # Handle case where k does not happen to comprise a border between two classes.
        # Three types of interesting observations: treatment observations with score < score[k],
        # control observations with score > score[k], and all observations with score == score[k].
        tot_observations = len(data_group)
        treatment_conversions = np.sum(
            data_class[(data_score > data_score[k - 1]) * data_group])
        treatment_observations = np.sum(
            (data_score > data_score[k - 1]) * data_group)
        control_conversions = np.sum(
            data_class[(data_score < data_score[k - 1]) * ~data_group])
        control_observations = np.sum(
            (data_score < data_score[k - 1]) * ~data_group)
        # Now we still need to count the observations where the score equals data_score[k]
        # This qpproach would remove the need for a sort.
        subset_group = data_group[data_score == data_score[k - 1]]
        subset_class = data_class[data_score == data_score[k - 1]]
        treatment_observations_in_subset = np.sum(subset_group)
        control_observations_in_subset = np.sum(~subset_group)
        observations_in_subset = len(subset_group)
        # Sanity check:
        assert observations_in_subset == treatment_observations_in_subset + control_observations_in_subset,\
            "Mismatch in counting observations in subset?!?"
        treatment_conversions_in_subset = np.sum(subset_class[subset_group])
        control_conversions_in_subset = np.sum(subset_class[~subset_group])
        # Split in subset corresponding to k
        j = k - np.sum(data_score > data_score[k - 1])
        treatment_conversions += j * treatment_conversions_in_subset /\
            observations_in_subset
        # Again, every observation in the subset with equal scores should be
        # treated as the "average observation" in the group!
        treatment_observations += j * np.sum(subset_group) / observations_in_subset
        control_conversions += (observations_in_subset - j) * control_conversions_in_subset /\
            observations_in_subset
        control_observations += (observations_in_subset - j) * \
            np.sum(~subset_group) / observations_in_subset
        treatment_conversion_rate = (treatment_conversions + smoothing) /\
            max(treatment_observations + 2 * smoothing, 1)
        control_conversion_rate = (control_conversions + smoothing) /\
            max(control_observations + 2 * smoothing, 1)
        conversion_rate = k / tot_observations * treatment_conversion_rate +\
            (tot_observations - k) / tot_observations * control_conversion_rate

    return conversion_rate


def expected_uplift(data_class, data_score, data_group, k=None,
                    model_2_score=None, model_2_k=None,
                    ref_plan_type=None,
                    smoothing=0):
    """
    Function for estimating expected uplift for a given treatment
    plan w.r.t a given reference plan, i.e. this is the difference
    between the conversion rate produced by the treatment plan and
    the conversion rate produced by some alternative (reference plan).
    The treatment plan is defined by data_score and k so that the 
    k highest scoring observations are treated. This is a point 
    estimate and if the reference plan is set to 'rand' this 
    corresponds to one point on the uplift curve. With 
    ref_plan_type = 'data', this is the formula presented in
    Gross & Tibshirani (2016).

    Parameters
    ----------
    data_class : np.array([bool])
        Array of labels for the observations. True indicates positive label.
    data_score : np.array([float])
        Array of uplift scores for the observations.
    data_group : np.array([bool])
        Array of group for the observations. True indicates that the observations is a treatment observation.
    k : int
        Number of observations that should be treated according to the treatment plan. If no
        value is provided it is assumed that all observations with positive uplift will be treated.
    model_2_score : np.array([float])
        Estimated uplift of observations for reference plan. Allows comparison of two uplift models.
        Can only be used in ref_plan_type == 'data'.
    model_2_k : int
        Number of observations that should be treated according to the reference treatment plan.
        Can be used only if ref_plan_type == 'model_2'
    ref_plan_type : str
        What method to use for estimation of the conversion rate for the reference plan.
        Has to be one of 
            'no_treatments' - no treatments applied,
            'all_treatments' - all observations are treated,
            'model_2' - top-model_2_k scoring observations following the model_2_score are treated.
            'gross' - as implemented by Gross & Tibshirani (2016)
    smoothing : float
        What value to use for smoothing.
    """

    if k is None:
        k = sum(data_score > 0)
    # Expected conversion rate for treatment plan with k treated observations:
    conversion_for_plan = expected_conversion_rate(
        data_class, data_score, data_group, k, smoothing)
    if ref_plan_type == 'no_treatments':
        conversion_for_ref = expected_conversion_rate(data_class, data_score,
                                                      data_group, 0, smoothing)
    elif ref_plan_type == 'all_treatments':
        conversion_for_ref = expected_conversion_rate(data_class, data_score, data_group,
                                                      len(data_group), smoothing)
    elif ref_plan_type == 'model_2':
        # "plan_b" another scoring plan with ref_k
        conversion_for_ref = expected_conversion_rate(data_class, model_2_score,
                                                      data_group, model_2_k, smoothing)
    elif ref_plan_type == 'gross':
        # Use treatments as used in the data for reference.
        # This was used by Gross & Tibshirani (2016) with smoothing = 0.
        conversion_for_ref = (sum(data_class) + smoothing) / \
            (len(data_class) + 2 * smoothing)

    tmp_uplift = conversion_for_plan - conversion_for_ref
    return tmp_uplift


@jit(nopython=True)
def _expected_conversion_rates(data_class, data_score, data_group,
                               smoothing=0):
    """
    This function estimates the expected conversion rates for all
    possible treatment rates in the data and returns a list of these.
    This list will have N+1 elements for conversion rates for treatment
    rate 0, 1/N, ... N/N, where N is the total number of observations.
    This is used e.g. to plot the uplift curve and estimate AUUC.

    Parameters
    ----------
    data_class : np.array([bool])
        Array of labels for the observations. True indicates positive label.
    data_score : np.array([float])
        Array of uplift scores for the observations.
    data_group : np.array([bool])
        Array of group for the observations. True indicates that the observations is a treatment observation.
    smoothing : float
        Smoothing used for estimation of conversion rates.
    """
    # Order all (this is used in the loop)
    n_observations = len(data_group)
    data_idx = np.argsort(data_score)[::-1]  # Sort in descending order.
    data_score = data_score[data_idx]
    data_group = data_group[data_idx]
    data_class = data_class[data_idx]

    # Initial counts: no treated observations
    conversions = []  # List of conversion rates for treatment plan and k
    treatment_goals = 0
    treatment_observations = 0
    control_goals = np.sum(data_class[~data_group])
    control_observations = np.sum(~data_group)
    # NUMERIC STABILITY?
    conversions.append((control_goals + smoothing) /
                       max(1, control_observations + 2 * smoothing))
    # Needs to be averaged at end.
    previous_score = np.finfo(np.float32).min
    k = 0  # Counter for how many observations are "treated" of N (n_observations)
    first_iteration = True
    tmp_treatment_observations = 0
    tmp_treatment_goals = 0
    tmp_control_observations = 0
    tmp_control_goals = 0
    tmp_n_observations = 0

    for item_class, item_score, item_group in zip(data_class, data_score, data_group):
        if item_score == previous_score:
            # Add items to counters
            # If item is 'treatment group'
            tmp_treatment_observations += int(item_group)
            tmp_treatment_goals += int(item_group) * item_class
            # If item is 'control group'
            tmp_control_observations += int(~item_group)
            tmp_control_goals += int(~item_group) * item_class
            tmp_n_observations += 1  # One more observation to handle.
        else:
            if not first_iteration:
                # Not first iteration. Handle all equally scoring observations.
                for i in range(1, tmp_n_observations + 1):  # 0-indexing...
                    # We want to treat every equally scoring observations as the average of
                    # all equally scoring observations (i.e. expectation).
                    # Remember that control observations should be
                    # subtracted from the total and treatment
                    # observations added.
                    # Fraction of observations our model would like to treat:
                    p_t = (k - tmp_n_observations + i) / n_observations
                    # Fraction of observations our model would _not_ like to treat:
                    p_c = (n_observations - k + tmp_n_observations - i) / n_observations

                    tmp_t_goals = treatment_goals + i * tmp_treatment_goals / tmp_n_observations
                    tmp_t_observations = treatment_observations + i * tmp_treatment_observations / tmp_n_observations
                    # max() is here only to deal with the case where there are zero observations. This
                    # corresponds to estimating the conversion rate of zero observations to zero.
                    tmp_t_conversion = (tmp_t_goals + smoothing) /\
                        max(1, tmp_t_observations + 2 * smoothing)

                    tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_observations
                    tmp_c_observations = control_observations - i * tmp_control_observations / tmp_n_observations
                    tmp_c_conversion = (tmp_c_goals + smoothing) /\
                        max(1, tmp_c_observations + 2 * smoothing)
                    # The expected conversion rate when the i first observations should be treated
                    # is a weighted average of the two above:
                    conversion_rate = p_t * tmp_t_conversion + p_c * tmp_c_conversion
                    conversions.append(conversion_rate)
                # Add all observations as integers to treatment and control counters (not tmp)
                treatment_goals += tmp_treatment_goals
                treatment_observations += tmp_treatment_observations
                control_goals -= tmp_control_goals
                control_observations -= tmp_control_observations

            # Reset counters and add new item
            # If item is 'treatment group'
            tmp_treatment_observations = int(item_group)
            tmp_treatment_goals = int(item_group) * item_class
            # If item is 'control group'
            tmp_control_observations = int(~item_group)
            tmp_control_goals = int(~item_group) * item_class
            tmp_n_observations = 1
            previous_score = item_score
            first_iteration = False
        k += 1
    # Handle last observations
    for i in range(1, tmp_n_observations + 1):  # 0-indexing...
        # Remember that control observations should be
        # subtracted from the total and treatment
        # observations added. goal == conversion... conversion -> conversion_rate
        # Fraction of observations our model would like to treat:
        p_t = (k - tmp_n_observations + i) / n_observations
        # Fraction of observations our model would _not_ like to treat:
        p_c = (n_observations - k + tmp_n_observations - i) / n_observations

        tmp_t_goals = treatment_goals + i * tmp_treatment_goals / tmp_n_observations
        tmp_t_observations = treatment_observations + i * tmp_treatment_observations / tmp_n_observations
        # max() is here only to deal with the case where there are zero observations. This
        # corresponds to estimating the conversion rate of zero observations to zero.
        tmp_t_conversion = tmp_t_goals / max(1, tmp_t_observations)

        tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_observations
        tmp_c_observations = control_observations - i * tmp_control_observations / tmp_n_observations
        tmp_t_conversion = (tmp_t_goals + smoothing) / \
            max(1, tmp_t_observations + 2 * smoothing)

        tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_observations
        tmp_c_observations = control_observations - i * tmp_control_observations / tmp_n_observations
        tmp_c_conversion = (tmp_c_goals + smoothing) / \
            max(1, tmp_c_observations + 2 * smoothing)
        # The expected conversion rate when the i first observations should be treated
        # is a weighted average of the two above:
        conversion_rate = p_t * tmp_t_conversion + p_c * tmp_c_conversion
        conversions.append(conversion_rate)
    return conversions


def auuc_metric(data_class, data_score, data_group,
                ref_plan_type='rand',
                smoothing=0,
                testing=False):
    """
    This function estimates the Area under the Uplift Curve, AUUC
    (Jaskowski & Jarosewicz, 2012). This is equivalent to estimating
    the expected uplift for some treatment plan for all treatment 
    rates (0, 1/N, ..., N/N) and averaging. This can be interpreted 
    as the expected conversion rate if you have no preference on 
    treatment rate. This function can also be used to estimate the 
    expected uplift compared to some other treatment plan than random
    targeting (as is assumed by AUUC; "uplift" for some treatment 
    rate is the improvement over randomly targeting the same fraction 
    of users).

    Parameters
    ----------
    data_class : np.array([bool])
        Array of labels for the observations. True indicates positive label.
    data_score : np.array([float])
        Array of uplift scores for the observations. Highest scoring observations are treated first.
    data_group : np.array([bool])
        Array of group for the observations. True indicates that the observations was treated.
    ref_plan_type : str
        Defines what reference plan should be used. Alternatives are:
        'rand' - Default value. This will result in estimation of AUUC,
        'all_treatments' - results in an estimate of average improvement in conversion rate 
         compared to applying treatments to all observations,
        'no_treatments' - results in an estimate of average improvement in conversion rate 
         compared to applying no treatments.
        'zero' - No reference plan. This results in estimation of the expected
         conversion rate given no preference for treatment rate (in contrast to AUUC
         that specifically focuses on the improvement in conversion rate, the uplift).
    smoothing : float
        Smoothing parameter to use for estimation of conversion rates.
    testing : bool
        A flag for testing purposes. If set to True, the function uses a
        different function for estimating expected conversion rates. This
        is extremely slow.

    Notes
    -----
    With "smoothing=0", this function estimates the conversion rate for
    treatment or conversion observations to zero if there are no observations to estimate
    from. This issue is negligible assuming that the number of observations N is large.
    The error introduced is smaller than 1/N.
    """

    if testing:
        warnings.warn("auuc_metric(testing=True) is for testing purposes only!" +
                      "The code is _very_ slow!")
        # Here for testing purposes!
        e_uplift = 0
        conversions = []
        n_observations = len(data_group)
        for i in range(n_observations + 1):
            conversion = expected_conversion_rate(data_class, data_score,
                                                  data_group, i, smoothing)
            conversions.append(conversion)
            e_uplift += 1 / (n_observations + 1) *\
                expected_uplift(data_class, data_score, data_group,
                                i, ref_plan_type=ref_plan_type)
    else:
        conversions = _expected_conversion_rates(data_class, data_score, data_group,
                                                 smoothing=smoothing)
        if ref_plan_type == 'no_treatments':
            ref_conversion = conversions[0]
            # Would be more efficient to do the subtraction only once.
            uplifts = [conversion -
                       ref_conversion for conversion in conversions]
        elif ref_plan_type == 'all_treatments':
            ref_conversion = conversions[-1]
            uplifts = [conversion -
                       ref_conversion for conversion in conversions]
        elif ref_plan_type == 'rand':
            conversion_0 = conversions[0]
            conversion_1 = conversions[-1]
            n_observations = len(data_class)
            # Simply subtracting the average of the two above
            # from the mean should be enough.
            uplifts = [conversion - i / n_observations * conversion_1 - (n_observations - i) /
                       n_observations * conversion_0
                       for i, conversion in zip(range(len(data_class) + 1), conversions)]
        elif ref_plan_type == 'zero':
            # This corresponds to estimating E_r over conversion rates
            uplifts = conversions
        else:
            raise Exception("Illegal ref_plan_type")
        # E_r is simply the mean of these uplifts:
        e_uplift = np.mean(uplifts)
    return e_uplift


@jit(nopython=True)
def bin_equally_scoring_observations(data_class, data_score):
    """
    Auxiliary function for kendalls_uplift_tau() below.
    Creates a list of bins where observations in one bin all 
    share the same score. Makes it easier to deal with 
    expectations etc. Both class and scores are changed 
    to floats (works better with numba).

    Parameters
    ----------
    data_class : np.array([bool])
        Array of labels for the observations. True indicates positive label.
    data_score : np.array([float])
        Array of uplift scores for the observations.
    """
    previous_score = data_score[0]
    tmp_class_sum = 0.0
    tmp_n = 0.0
    score_distribution = []
    for item_score, item_class in zip(data_score, data_class):
        if item_score == previous_score:
            # This is always true for first item
            tmp_class_sum += float(int(item_class))
            tmp_n += 1
        else:
            # Score changed. Append items to list
            # Lists of dicts are not supported by numba.
            score_distribution.append([previous_score,
                                       tmp_n,
                                       previous_score * tmp_n,
                                       tmp_class_sum,
                                       tmp_class_sum / tmp_n])
            # Reset flags and counters and add latest observation
            previous_score = item_score
            tmp_n = 1.0
            tmp_class_sum = float(int(item_class))
    # Handling last item:
    score_distribution += [[previous_score,
                               tmp_n,
                               previous_score * tmp_n,
                               tmp_class_sum,
                               tmp_class_sum / tmp_n]]
    return score_distribution


@jit(nopython=True)
def kendalls_uplift_tau(data_class,
                        data_score,
                        data_group,
                        k=100,
                        tolerance=1e-6):
    """
    Function for estimating Kendall's tau (i.e. rank _correlation_
    between k bins). This version handles ties by averaging equally
    scoring observations over neighboring bins (i.e. treating the 
    observations as an expectation of all equally scoring observations
    in that group).

    Parameters
    ----------
    data_class : np.array([bool]) 
        Array of class-labels. True indicates positive observation.
    data_score : np.array([float]) 
        Array of scores for observations. These are assumed to be strictly 
        monotonically related to the uplift estimates of the observations.
    data_group : np.array(bool) 
        True indicates treatment group, False control group.
    k : int 
        Number of bins. Belbahri (2019, Arxiv) suggested using 5 or 10, 
        which is just way too few for larger datasets.
    tolerance : float 
        Acceptable error due to machine accuracy causing small discrepancies.

    Notes:
    -This version got very ugly as numba does not support lists of dicts.
    The relevant list-indices to keep in mind are
    {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}

    -This version uses the group (treatment or control) with fewer observations
    to set bin boundaries. The point with this is to ensure that all bins
    get a more even number of observations in the group with less observations to
    get more narrow credible intervals. Using the majority group for
    setting bin boundaries could also be a good idea as that would result
    in boundaries being in more "correct" positions. It is not clear which
    version should be better. This could perhaps be analyzed empirically.
    -Further the bins are populated so that if k=100 and there are 150 observations
    in the minority group, all bins get 1.5 observations. The fractions are
    split into bins as weighted expectations of that observations (i.e. an observation
    that is split .3:.7 acts as .3 of one entire observation in one bin and
    as .7 of one observations in the other).
    -Equally scoring observations in one group are all treated as expectations
    of the observations (i.e. as an "average observation").
    -On a sidenote, due to using the same bin boundaries for the majority
    observations as defined by the minority observations, the majority observations in
    one bin will be slightly biased upwards as there may always be majority
    observations in one bin that scores in [max(min_score_i), min(min_score_i+1)]
    where i denotes *this bin. This bias is assumed to be tiny.
    -There is some numeric instability in this approach as we split observations
    etc. This is dealt with using tolerance.
    -Ties contribute 0 to the numerator of the correlation,
    but 1 to the denominator

    The simple version where we simply select boundaries and then place
    observations based on these would fail on Criteo data as there is a large
    number of observations with identical scores in the dataset (identical
    profiles, probably users Criteo knew nothing of). Hence this more
    complex version.
    """

    # Sort all data
    data_idx = np.argsort(data_score)
    data_score = data_score[data_idx]
    data_class = data_class[data_idx]
    data_group = data_group[data_idx]

    if data_score[0] == data_score[-1]:
        # All observations have equal score.
        return 0
    # True data_group equals treatment group, false equals control:
    treatment_n = np.sum(data_group)
    control_n = np.sum(~data_group)
    # Is the treatment or control group smaller?
    if treatment_n <= control_n:
        minority_group = 'treatment'
    else:
        minority_group = 'control'

    # Set the smaller group as minority group.
    # The class boundaries are set based on the smaller group
    # to ensure a more even distribution of observations in bins.
    if minority_group == 'treatment':
        # Then use the treatment group to set bin boundaries.
        min_score = data_score[data_group]
        min_class = data_class[data_group]
        maj_score = data_score[~data_group]
        maj_class = data_class[~data_group]
    else:
        min_score = data_score[~data_group]
        min_class = data_class[~data_group]
        maj_score = data_score[data_group]
        maj_class = data_class[data_group]

    min_score_distribution = \
        bin_equally_scoring_observations(min_class, min_score)
    maj_score_distribution = \
        bin_equally_scoring_observations(maj_class, maj_score)

    observations_per_bin = len(min_score) / k
    # Initialize bin parameters for every bin:
    min_bins = [[0.0, 0.0, 0.0, 0.0, 0.0] for tmp in range(k)]
    maj_bins = [[0.0, 0.0, 0.0, 0.0, 0.0] for tmp in range(k)]
    # Initialize counters:
    i = 0  # Bin counter for result
    j = 0  # Bin counter for treatment_score_distribution
    l = 0  # Bin counter for control_score_distribution
    # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
    tmp_min_bin = [0.0, 0.0, 0.0, 0.0]  # scores, observations, score_sum, and positives
    tmp_maj_bin = [0.0, 0.0, 0.0, 0.0]
    # In this while-loop, the boundaries and fractions of minority
    # observations should be recorded to treat the majority observations equally.
    while i < k:  # Where 'k' is the desired number of bins.
        # I.e. while there are still bins left to populate
        # Majority class: if score in bin l smaller than in j,
        # add all observations to maj_bins[i]
        # Majority observations
        # don't necessarily exhibit the same scores as the minority observations.
        # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
        while min_bins[i][1] < (observations_per_bin - tolerance):
            # While *this bin is not full
            # First take observations from tmp_min_bin if there are any:
            # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
            if tmp_min_bin[1] <= observations_per_bin and tmp_min_bin[1] != 0:
                # Just put all observations in bins[i]:
                min_bins[i][1] += tmp_min_bin[1]
                min_bins[i][3] += tmp_min_bin[3]
                min_bins[i][2] += tmp_min_bin[2]
                # Empty tmp_min_bin:
                tmp_min_bin[1] = 0
                tmp_min_bin[3] = 0
                tmp_min_bin[2] = 0
                # Place all equally scoring maj-observations in bin i:
                maj_bins[i][1] += tmp_maj_bin[1]
                maj_bins[i][3] += tmp_maj_bin[3]
                maj_bins[i][2] += tmp_maj_bin[2]
                # Reset tmp-bin:
                tmp_maj_bin[1] = 0
                tmp_maj_bin[3] = 0
                tmp_maj_bin[2] = 0
                # First iteration: loop enters this if, the next elif
                # is not run because of it. That's fine.
                # +-tolerance to deal with numeric instability:
                if observations_per_bin - tolerance < min_bins[i][1] < observations_per_bin + tolerance:
                    # Bin i is full. Jump to next.
                    # At break, i is increased by one and we start populating the next bin.
                    break
            elif tmp_min_bin[1] > observations_per_bin:
                # That is, when there are more observations in tmp_min_bin
                # than should fit in one bin, fill the bin...
                # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                min_bins[i][1] = observations_per_bin  # If min bins was not empty, this will cause issues?!?
                # tmp_min_bin is always a carry over of observations that did not fit in previous bin
                # and in this elif clause, there are more observations in tmp than will fit in *this bin also.
                min_bins[i][3] += observations_per_bin /\
                    tmp_min_bin[1] * tmp_min_bin[3]
                min_bins[i][2] += observations_per_bin /\
                    tmp_min_bin[1] * tmp_min_bin[2]
                # Estimate the fraction before removing observations from tmp_min_bin:
                # This fraction of maj observations should also be put in teh bin.
                frac = observations_per_bin / tmp_min_bin[1]
                # ...and then remove the observations from tmp_min_bin:
                tmp_min_bin[1] -= min_bins[i][1]
                tmp_min_bin[3] -= min_bins[i][3]
                tmp_min_bin[2] -= min_bins[i][2]
                # Next take a corresponding amount of maj-observations and add
                # to maj_bins[i]:
                tmp_maj_observations = frac * tmp_maj_bin[1]
                tmp_maj_positives = frac * tmp_maj_bin[3]
                tmp_maj_score_sum = frac * tmp_maj_bin[2]
                # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                maj_bins[i][1] = tmp_maj_observations  # THERE SHOULD NOT BE ANY in this maj_bin from previously.
                maj_bins[i][3] = tmp_maj_positives
                maj_bins[i][2] = tmp_maj_score_sum
                # Leave in tmp-bins what has not been placed yet.
                tmp_maj_bin[1] -= tmp_maj_observations
                tmp_maj_bin[3] -= tmp_maj_positives
                tmp_maj_bin[2] -= tmp_maj_score_sum
                break  # Necessary here! Will break out of inner
                # loop and add one to i (and then run the while condition).
                # We just filled the previous bin.
            # The score boundaries are set by j. If j exceeds len(min...)
            # that must be because of numeric instability (i.e. last bin
            # is not full).
            # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
            if min_score_distribution[j][1] <=\
                observations_per_bin - min_bins[i][1]:
                # If number of observations in j is smaller or equal to the
                # number of observations that fit in bins[i]:
                min_bins[i][1] += min_score_distribution[j][1]
                min_bins[i][3] += min_score_distribution[j][3]
                min_bins[i][2] += min_score_distribution[j][2]
                while l < len(maj_score_distribution):
                    # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                    # While score in maj_... is smaller or equal to the corresponding min bin:
                    if maj_score_distribution[l][0] <=\
                        min_score_distribution[j][0]:  # pytest.approx here? Scores should not be unstable.
                        # If all min observations with *this
                        maj_bins[i][1] += maj_score_distribution[l][1]
                        maj_bins[i][3] += maj_score_distribution[l][3]
                        maj_bins[i][2] += maj_score_distribution[l][2]
                    else:
                        # l remains for next iteration.
                        break
                    # Try all bins until criterion above is not satisfied.
                    l += 1
                # We just moved _all_ observations from *this distribution bin
                # to bins[i]. Move on.
                # All observations fit in *this bin.
            else:  # There are more observations in min_sc...[.] than will fit in this bin:
                # Hypothetically (without additional break above)
                # we could at this point be in a case where min_bins is full
                # and we are merely trying to move around zero observations.
                # (i.e. tmp_n == 0)
                # 
                # Let's take just the number of observations from j that
                # fit into bin[i]:
                # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                tmp_n = observations_per_bin - min_bins[i][1]  # number of observations to move to *this bin
                frac = tmp_n / min_score_distribution[j][1]  # Fraction of observations to take from bin
                min_bins[i][3] += frac * min_score_distribution[j][3]
                min_bins[i][1] += tmp_n
                min_bins[i][2] += frac * min_score_distribution[j][2]
                # Now bins[i]['observations'] should equal observations_per_bin.
                # Sanity check:
                #assert observations_per_bin == pytest.approx(min_bins[i]['observations'])
                # Carry over remaining observations
                # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                tmp_min_bin = [0,  # score, not used
                               min_score_distribution[j][1] - tmp_n,  # observations
                               min_score_distribution[j][2] * (1 - frac),
                               min_score_distribution[j][3] * (1 - frac)]
                while l < len(maj_score_distribution):
                    # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                    if maj_score_distribution[l][0] <\
                        min_score_distribution[j][0]:
                        # Put all observations in bin:
                        maj_bins[i][1] += maj_score_distribution[l][1]
                        maj_bins[i][3] += maj_score_distribution[l][3]
                        maj_bins[i][2] += maj_score_distribution[l][2]
                    elif maj_score_distribution[l][0] ==\
                        min_score_distribution[j][0]:
                        # All observations in min group did not fit in
                        # *this bin. Now we want the same fraction
                        # of maj observations into maj_bins[i].
                        # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                        tmp_maj_observations = frac * maj_score_distribution[l][1]
                        tmp_maj_positives = frac * maj_score_distribution[l][3]
                        tmp_maj_score_sum = frac * maj_score_distribution[l][2]
                        maj_bins[i][1] += tmp_maj_observations
                        maj_bins[i][3] += tmp_maj_positives
                        maj_bins[i][2] += tmp_maj_score_sum
                        # Carry over remaining observations:
                        # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
                        tmp_maj_bin = [0,  # score, not used
                                       maj_score_distribution[l][1] -  # observations
                                       tmp_maj_observations,
                                       maj_score_distribution[l][2] -  # score_sum
                                       tmp_maj_score_sum,
                                       maj_score_distribution[l][3] -  # positives
                                       tmp_maj_positives]
                    else:
                        # maj_bin[l]['score'] is higher than in min_bin[j]
                        # which is currently being treated:
                        break
                    l += 1

                # All observations from bin j moved to tmp_min_bin.
            j += 1
        # Move on to next bin:
        i += 1
    # If the maj_bins still have observations that are untreated when min_bins is full,
    # those need to be dealt with separately:
    while l < len(maj_score_distribution):
        # key is {'score': 0, 'observations': 1, 'score_sum': 2, 'positives': 3, 'expected_class': 4}
        maj_bins[-1][1] += maj_score_distribution[l][1]
        maj_bins[-1][2] += maj_score_distribution[l][2]
        maj_bins[-1][3] += maj_score_distribution[l][3]
        l += 1

    # Sanity check (number of observations assigned to lists equals number of input observations):
    sum_min_observations = np.sum(np.array([item[1] for item in min_bins]))
    sum_maj_observations = np.sum(np.array([item[1] for item in maj_bins]))

    n_observations = len(data_class)
    checksum = sum_min_observations + sum_maj_observations
    if (n_observations - tolerance) < checksum < n_observations + tolerance:
        # All is fine
        pass
    else:
        print("Error: Not dealing with all observations.")
        # Numba does not handle string handling to print more info here.
        return np.nan
    # Just split one group into k bins and treat equally scoring
    # observations as the expectation of every equally scoring observation and place those
    # in the same bins.
    # The scores now represent the uplift estimates and we are only interested in
    # the ranks between observations. Due to the composition, the bins should have a
    # strictly monotonically increasing scores.
    # Next do the correlation estimation. equal score and equal uplift estimate
    # equals '0', bins in right order +1, bins in wrong order -1, normalize
    # by #comparisons (N*(N-1)).
    if minority_group == 'treatment':
        treatment_bins = min_bins
        control_bins = maj_bins
    else:
        treatment_bins = maj_bins
        control_bins = min_bins
    uplifts = []  # item[0] is uplift, item[1] is score
    for treatment, control in zip(treatment_bins,
                                  control_bins):
        # Due to the bin boundaries being set from one group (minority observations)
        # there is not a guarantee that all bins get majority observations. In such
        # cases Kendall's uplift tau cannot be estimated.
        # Actually, asserts should be replaced by "return NaN and print text above."
        #if treatment['observations'] == 0 or control['observations'] == 0:
        if treatment[1] == 0 or control[1] == 0:
            # No observations in bin. Tau cannot be estimated.
            print("Cannot estimate Kendall's uplift tau with given k.")
            print("(Bin with zero observations.) Perhaps try smaller k.")
            return np.nan
        p_t = treatment[3] / treatment[1]
        p_c = control[3] / control[1]
        # This is a kind of average that uses the fact that
        # scores must be monotonic with bin index, i.e. scores
        # does not have to map linearly to probabilities.
        bin_score = (treatment[2] + control[2]) /\
            (treatment[1] + control[1])
        uplifts.append([p_t - p_c, bin_score])
    # Remember that 'k' is the number of bins!
    tmp_corr = 0
    for i in range(k - 1):
        # Bins with identical scores contribute +1 if the
        # uplift estimates also are identical, -1 otherwise.
        # Using pytest.approx to deal with numeric instability.
        for j in range(i+1, k):
            if uplifts[j][0] - tolerance < uplifts[i][0] < uplifts[j][0] + tolerance:
                # Using pytest.approx to deal with numeric instability.
                if uplifts[j][1] - tolerance < uplifts[i][1] < uplifts[j][1] + tolerance:
                    # This is a tie. While it could be argued that the bins
                    # are in the "correct" order, having multiple bins with
                    # equal scores is not useful, although not detrimental
                    # either. Hence the += 0.
                    tmp_corr += 0
                else:
                    tmp_corr -= 1
            elif uplifts[i][0] < uplifts[j][0]:
                if uplifts[i][1] < uplifts[j][1]:
                    tmp_corr += 1
                else:
                    # Equal or larger
                    tmp_corr -= 1
            elif uplifts[i][0] > uplifts[j][0]:
                if uplifts[i][1] > uplifts[j][1]:
                    tmp_corr += 1
                else:
                    tmp_corr -= 1
            else:
                raise Exception("Estimation of correlation failed")
    # Return estimated Kendall's uplift tau:
    result = (tmp_corr / (k * (k - 1) / 2))
    return result


@jit(nopython=True)
def _qini_points(data_class,
                 data_score,
                 data_group):
    """Auxiliary function for qini_coefficient(). Returns the
    points on the qini-curve.

    Parameters
    ----------
    data_class : numpy.array([bool])
    data_score : numpy.array([float])
    data_group : numpy.array([bool]) 
        True indicates that observation belongs to the treatment-group.
    """
    # Order data in descending order:
    data_idx = np.argsort(data_score)[::-1]
    data_class = data_class[data_idx]
    data_score = data_score[data_idx]
    data_group = data_group[data_idx]

    # Set initial values for counters etc:
    qini_points = []
    # Normalization factor (N_t / N_c):
    n_factor = np.sum(data_group) / np.sum(~data_group)
    control_goals = 0
    treatment_goals = 0
    score_previous = np.finfo(np.float32).min
    tmp_n_observations = 1  # Set to one to allow division in first iteration
    tmp_treatment_goals = 0
    tmp_control_goals = 0
    for item_class, item_score, item_group in\
            zip(data_class, data_score, data_group):
        if score_previous != item_score:
            # Skip this section until we find the next observation with differing score.
            # If we have a 'new score', handle the observations
            # currently stored as counts...
            for i in range(1, tmp_n_observations + 1):
                # Tie handling. Generate observations by interpolation.
                # This is equivalent to drawing a straight line on the
                # qini curve between the previous and the consequtive point.
                tmp_qini_point = (treatment_goals + i * tmp_treatment_goals /
                                  tmp_n_observations) -\
                    (control_goals + i * tmp_control_goals /
                     tmp_n_observations) * n_factor
                qini_points.append(tmp_qini_point)
            # Add tmp items to vectors before resetting them
            treatment_goals += tmp_treatment_goals
            control_goals += tmp_control_goals
            # Reset counters
            tmp_n_observations = 0
            tmp_treatment_goals = 0
            tmp_control_goals = 0
            score_previous = item_score
        # Add item to counters:
        tmp_n_observations += 1
        tmp_treatment_goals += int(item_group) * item_class
        tmp_control_goals += int(~item_group) * item_class

    # Handle remaining observations:
    for i in range(1, tmp_n_observations + 1):
        tmp_qini_point = (treatment_goals + i * tmp_treatment_goals /
                          tmp_n_observations) -\
            (control_goals + i * tmp_control_goals /
             tmp_n_observations) * n_factor
        qini_points += [tmp_qini_point]  # Numba fix (it has issues with appending here using .append()).

    # Make list into np.array:
    qini_points = np.array(qini_points)
    return qini_points


@jit(nopython=True)
def qini_coefficient(data_class, data_score, data_group):
    """Function for calculating the qini-coefficient of some data.
    This version follows Radcliffe (2007), which is the original
    source for the metric. This function implements tie handling.
    
    Parameters
    ----------
    data_class : numpy.array([bool])
    data_score : numpy.array([float])
    data_group : numpy.array([bool])
    """
    qini_points = _qini_points(data_class, data_score, data_group)
    numerator = np.sum(qini_points)

    # Create artificial "optimal" ordering (that maximizes this
    # function) to estimate the denominator.
    new_data_group = np.array(([True] * np.sum(data_group)) +
                              ([False] * np.sum(~data_group)))

    new_data_class = np.array(([True] * np.sum(data_class[data_group])) +
                              ([False] * np.sum(~data_class[data_group])) +
                              ([False] * int(np.sum(~data_class[~data_group]))) +
                              ([True] * int(np.sum(data_class[~data_group]))))

    # Score array so that first observation have highest score. This is just to get
    # the sorting not to change the order for estimation of "optimal ordering."
    new_data_score = np.array([i for i in range(len(data_group))][::-1])
    new_qini_points = _qini_points(data_class=new_data_class,
                                   data_score=new_data_score,
                                   data_group=new_data_group)
    denominator = np.sum(new_qini_points)
    # Calculate the qini-coefficient:
    result = numerator / denominator
    return result


@jit(nopython=True)
def _euce_points(data_class, data_prob, data_group,
                 k=100):
    """Auxiliary function for expected_uplift_calibration_error().
    This one is numba-optimized. This could also be used for visualization.

    Parameters
    ----------
    data_class : numpy.array([bool])
    data_prob : numpy.array([float]) 
        Predicted change in conversion probability for each observation.
    data_group : numpy.array([bool])
    k : int 
        Number of groups to split the data into for estimation.
    """
    # Doesn't matter if the sorting is ascending or descending.
    idx = np.argsort(data_prob)
    n_observations = len(data_prob)
    expected_errors = []
    # data_class = np.array([bool(item) for item in data_class])
    for i in range(k):
        tmp_idx = idx[int(n_observations / k * i):int((1 + i) * n_observations / k)]
        treatment_goals = np.sum(data_class[tmp_idx][data_group[tmp_idx]])
        treatment_observations = np.sum(data_group[tmp_idx])
        control_goals = np.sum(data_class[tmp_idx][~data_group[tmp_idx]])
        control_observations = np.sum(~data_group[tmp_idx])
        # Sanity check:
        assert treatment_observations + control_observations == len(tmp_idx), \
            "Error in estimation of expected calibration rate"
        assert treatment_goals + control_goals == np.sum(data_class[tmp_idx]),\
            "Error in estimation of expected calibration rate"
        uplift_in_data = (treatment_goals / treatment_observations) - \
            (control_goals / control_observations)
        estimated_uplift = np.mean(data_prob[tmp_idx])
        expected_errors.append(np.abs(uplift_in_data - estimated_uplift))

    # Make numba-compatible:
    return np.array(expected_errors)


def expected_uplift_calibration_error(data_class, data_prob, data_group,
                                      k=100, verbose=False):
    """
    Function for estimating the expected calibration error and maximum
    calibration error for uplift. This is an extension of the ECE and MCE
    presented by Naeini & al. in 2015 (their metrics focused on response
    calibration, ours on uplift calibration).

    Parameters
    ----------
    data_class : numpy.array([bool])
    data_prob : numpy.array([float]) 
        Predicted change in conversion probability for each observation.
    data_group : numpy.array([bool])
    k : int 
        Number of groups to split the data into for estimation.
    """

    # Sanity check
    if k > len(data_class):
        raise Exception("k needs to be smaller than N!")

    try:
        expected_errors = _euce_points(data_class, data_prob,
                                    data_group, k=k)
    except Exception as e:  # This exception should be made _much_ more selective. Should.
        print("******************************************")
        print("ERROR: Failed to run uplift_metrics._euce_points: %s" % e)
        print("******************************************")
        expected_errors = [float("nan"), float("nan")]
    euce = np.mean(expected_errors)
    muce = np.max(expected_errors)
    if verbose:
        print("Expected uplift calibration error: {}".format(euce))
        print("Maximum uplift calibration error: {}".format(muce))
    return (euce, muce)


def estimate_adjusted_e_mse(data_class, data_score, data_group):
    """
    Function for estimating expectation of mean squared
    error plus constant. The constant is fixed for a dataset
    but unknowable. As a consequence, E(MSE) + C is a valid
    metric for goodness of fit e.g. for model comparisons.
    
    Parameters
    ----------
    data_class : np.array 
        Classes of testing observations
    data_score : np.array 
        The predicted uplift. Note that the value matters in this 
        metric (in contrast to e.g. AUUC where only rank matters).
    data_group : np.array 
        The group of the observations.
    """
    # 1. Calculate the revert-label from the data. Hypothetically
    # we could also request that the function is passed the already
    # estimated values r from the DatasetCollection.
    # -Which approach is more correct?
    # -A small testing set will create r-values that are unstable
    # -Is it correct to use r-values estimated from a bigger dataset?
    # I would think that it is most correct to estimate directly form
    # data at hand.
    N_t = sum(data_group == True)
    N_c = sum(data_group == False)
    # Sanity check:
    assert N_t + N_c == data_group.shape[0], "Error in observation count (_revet_label())."
    # This needs to be sorted out.
    p_t = N_t / (N_t + N_c)
    assert 0.0 < p_t < 1.0, "Revert-label cannot be estimated from only t or c observations."
    def revert(y_i, t_i, p_t):
        return (y_i * (int(t_i) - p_t) / (p_t * (1 - p_t)))
    r_vec = np.array([revert(y_i, t_i, p_t) for y_i, t_i in zip(data_class, data_group)])
    #r_vec = r_vec.astype(self.data_format['data_type'])
    # Now we have data_score and r_vec to estimate adjusted E(MSE) from.
    emse = np.mean([(item_1 - item_2)**2 for item_1, item_2 in zip(data_score, r_vec)])
    return emse


def beta_difference_uncertainty(alpha1, beta1, alpha0, beta0,
                     prior_a1=1, prior_b1=1,
                     prior_a0=1, prior_b0=1,
                     N=100000,
                     p_mass=0.95):
    """
    This is a Monte Carlo (MC) approach to estimating
    uncertainty of a beta-difference distribution
    (Pham-Gia & Turkkan, 1993). The uncertainty is
    quantified as the highest posterior density credible
    interval (HPD-inverval). 

    The beta-difference distribution is equivalent to
    the uncertainty distribution of the difference between
    two Bernoulli-distributed variables. In the case of 
    uplift this corresponds to alpha1 and beta1 for
    :math:`p(y=1|x, do(t=1))` and alpha0 and beta 0 for
    :math:`p(y=1|x, do(t=0))`.

    Parameters
    ----------
    alpha1 : float 
        Alpha for distribution of :math:`p(y=1|x, do(t=1))`
    beta1 : float 
        Beta for distribution of :math:`p(y=1|x, do(t=1))`
    alpha0 : float
        Alpha for distribution of :math:`p(y=1|x, do(t=0))`
    beta0 : float 
        Beta for distribution of :math:`p(y=1|x, do(t=0))`
    prior_a1 : float 
        Prior for alpha1
    etc.
    N : int
        Number of observations to draw. Should probably be at least 10,000.
    p_mass : float
        In [0, 1]. The probability mass required inside of
        the interval.
    """
    # Draw observations from distribution
    p_t1 = beta.rvs(alpha1 + prior_a1, beta1 + prior_b1, size=N)
    p_t0 = beta.rvs(alpha0 + prior_a0, beta0 + prior_b0, size=N)
    tau = p_t1 - p_t0
    # Estimate HPD (95%).
    # If we defined the CDF as sum_i^j(tau_i) for
    # i <= j and j \in {0:N}, then we can on the ordered
    # observations do a simple search over all applicable
    # intervals and pick the shortest one. The number of
    # observations in one interval is
    # simply p * N where a typical
    # value for p would be 0.95.
    # 1. Sort tau in increasing order
    tau = np.sort(tau)
    # 2. Calculate window size N_{1-alpha}
    N = len(tau)
    n_interval = int(N * p_mass)
    lower_idx = 0
    upper_idx = n_interval
    # 3. Estimate width of sliding window
    smallest_width = np.inf
    while upper_idx < N:
        # We are only looking for any interval that
        # contains at least 95% of the observations. Any sliding window
        # containing this will do. If they additionally contain other
        # observations, that is fine.
        tmp_width = tau[upper_idx] - tau[lower_idx]
        if tmp_width < smallest_width:
            # Store results:
            # What is this has not been accessed at all? Should not be possible...
            smallest_width = tmp_width
            smallest_low_idx = lower_idx
            smallest_up_idx = upper_idx
        lower_idx += 1
        upper_idx += 1
    # 4. Pick narrowest.
    return {'width': smallest_width, 
            'lower_bound': tau[smallest_low_idx],
            'upper_bound': tau[smallest_up_idx]}  # This is not always set (?!?)


def test_for_beta_difference(alpha11, beta11, alpha12, beta12,
                             alpha21, beta21, alpha22, beta22,
                             prior_a11=1, prior_b11=1,
                             prior_a12=1, prior_b12=1,
                             prior_a21=1, prior_b21=1,
                             prior_a22=1, prior_b22=1,
                             N=100000):
    """
    Bayesian estimate for :math:`p(\tau(x_1)) > p(\tau(x_2))`,
    i.e. the probability that one uplift estimate is greater
    than the other. This is based on the assumption that the
    uncertainty of the uplift follows a beta-difference
    distribution (Pham-Gia & Turkkan, 1993).

    This is equivalent to assuming that conversion probabilities used 
    to estimate uplift are Bernoulli-distributed and that the parameters 
    for these follow Beta-distributions.

    Parameters
    ----------
    alpha11 : float
        Alpha for p(y=1|x, t=1) for tau_1
    beta11 : float
        Beta for p(y=1|x, t=1) for tau_1
    alpha12 : float
        alpha for p(y=1|x, t=0) for tau_1
    beta12 : float
        beta for p(y=1|x, t=0) for tau_1
    alpha21 : float
        alpha for p(y=1|x, t=1) for tau_2
    etc.
    prior_a11 : float 
        Alpha prior for p(y=1|x, t=1) for tau_1
    etc.
    N : int 
        Number of observations to draw. Should probably be at least 10,000.
    """
    p_t11 = beta.rvs(alpha11 + prior_a11, beta11 + prior_b11, size=N)
    p_t10 = beta.rvs(alpha12 + prior_a12, beta12 + prior_b12, size=N)
    tau_1 = p_t11 - p_t10
    p_t21 = beta.rvs(alpha21 + prior_a21, beta21 + prior_b21, size=N)
    p_t20 = beta.rvs(alpha22 + prior_a22, beta22 + prior_b22, size=N)
    tau_2 = p_t21 - p_t20

    # Probability that tau_1 > tau_2:
    n_pos = sum(tau_1 > tau_2)
    prob = n_pos / N
    return prob


def test_for_differences_in_mean(N_t1, N_c1,
                                 k_t1, n_t1, k_c1, n_c1,
                                 N_t2, N_c2,
                                 k_t2, n_t2, k_c2, n_c2,
                                 size=100000):
    """
    Bayesian test for differences in mean for uplift modeling.
    Basically this is a test for whether two treatment models produce
    conversion rates that are different to a statistically significant
    degree. This is similar to a bayesian test for difference in conversion
    rates where E(p_1 > p_2), i.e. an integral, is estimated using Monte
    Carlo simulation.
    The uncertainty for the conversion rate for a model is characterized
    by three beta-distributions:
    -one that characterizes the uncertainty of the treatment rate,
    -a second one charaterizes the uncertainty of the conversion rate
    for the treated observations, and
    -a third one characterizes the uncertainty of the conversion rate
    for the untreated (control) observations.

    Similar to test_for_beta_difference with uninformative priors.

    Parameters
    ----------
    N_t1 : int 
        Number of observations that model_1 would like to treat.
    N_c1 : int 
        Number of observations that model_1 would _not_ like to treat.
    k_t1 : int 
        Number of positive treatment observations that the treatment plan would have targeted.
    n_t1 : int 
        Number of treatment observations that the model_1 would have targeted.
    k_c1 : int 
        Number of observations below targeting threshold that ended up
        converting (i.e. positive control observations that model_1 would not
        have targeted).
    n_c1 : int 
        Number of control observations that model_1 would not have targeted.
    *2 : (*) 
        Similar as above, but for model_2.

    Notes:
    This function cannot be Numba-optimized as numba does not support Scipy.
    "model_x" can also be thought of as "treatment plan x" (e.g. vocabulary by
     Gross & Tibshirani, 2016).
    """
    # First model:
    observations_1_1 = beta.rvs(N_t1 + 1, N_c1 + 1, size=size)
    observations_1_2 = beta.rvs(k_t1 + 1, n_t1 - k_t1 + 1, size=size)
    observations_1_3 = beta.rvs(k_c1 + 1, n_c1 - k_c1 + 1, size=size)
    # Itemwise product and sum ('1' is recycled).
    tmp_1 = observations_1_1 * observations_1_2 + (1 - observations_1_1) * observations_1_3
    # Second model:
    observations_2_1 = beta.rvs(N_t2 + 1, N_c2 + 1, size=size)
    observations_2_2 = beta.rvs(k_t2 + 1, n_t2 - k_t2 + 1, size=size)
    observations_2_3 = beta.rvs(k_c2 + 1, n_c2 - k_c2 + 1, size=size)
    tmp_2 = observations_2_1 * observations_2_2 + (1 - observations_2_1) * observations_2_3
    prob = sum(tmp_1 > tmp_2) / float(size)
    return prob
