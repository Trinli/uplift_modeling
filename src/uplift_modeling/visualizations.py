"""
Functions for uplift-related plots.
"""
import matplotlib.pyplot as plt
from uplift_modeling.metrics import *


def plot_conversion_rates(data_class, data_score, data_group, file_name='conversion.png'):
    """Function for plotting conversion rates vs. treatment rates for treatment
    rate in [0, 1]. This is almost equivalent to the uplift-curve defined by
    Jaskowski & Jarosewicz (2012) with the difference that they subtract the
    baseline conversion (conversion with no treatments) from all conversion
    rates. The plot has the same shape as the uplift-curve, only the y-axis
    differs.

    Args:
    ...
    file_name (str): Name of the file where you want the plot stored. Include
     .png-suffix.
    """
    conversions = _expected_conversion_rates(
        data_class, data_score, data_group)
    # Plot conversion rates:
    plt.plot([100 * x / len(conversions) for x in range(0, len(conversions))],
             [100 * item for item in conversions])
    # Add random line:
    plt.plot([0, 100], [100 * conversions[0], 100 * conversions[-1]])
    plt.xlabel('Fraction of treated samples [%]')
    plt.ylabel('Conversion rate [%]')
    plt.title('Conversion rate vs. treatment rate')
    plt.savefig(file_name)
    plt.close()
    return()


def plot_uplift_curve(data_class, data_score, data_group, file_name='uplift_curve.png', revenue=None):
    """Function for plotting uplift vs. treatment rate following Jaskowski &
    Jarosewicz (2012). Very similar to plot_conversion_rates().
    This might be preferable if you want to highlight the change rather than the
    complete picture. E.g. if your model increases E_r(conversion rate) from
    3.1% to 3.15%, this is hardly relevant for the use case at hand. However
    if you are doing algorithm development, this difference might be interesting
    and then you might be better off studying uplift rather than conversion rates.

    Args:
    ...
    file_name (str): Name of the file where you want your uplift curve stored.
    revenue (float): Estimated revenue of one conversion. If 'revenue' is not
     None and is a floating point number, it will be used to estimate the
     incremental revenues of the uplift model. Otherwise the uplift curve will
     plot the increase in conversion rate as function of treatment rate. This
     is the default functionality.

    """
    conversions = _expected_conversion_rates(
        data_class, data_score, data_group)
    n_splits = len(conversions)
    # Conversion rate with no treatments
    conversion_0 = conversions[0]
    # Conversion rate with only treatments
    conversion_1 = conversions[-1]
    if revenue is not None:
        tmp = revenue
    else:
        tmp = 1
    plt.plot([100 * x / n_splits for x in range(0, n_splits)],
             [tmp * 100 * (conversion - conversion_0) for conversion, x in
              zip(conversions, range(len(conversions)))], color='tab:blue')  #, label='Tree')
    # Add line for "random" model:
    plt.plot([0, 100], [0, tmp * 100 * (conversion_1 - conversion_0)], color='tab:green')  #, label='(random)')
    if revenue is not None:
        plt.ylabel('Cumulative revenue increase')
    else:
        plt.ylabel('Uplift [%]')
    #plt.title('Uplift curve')
    plt.xlabel('Fraction of treated samples [%]')
    #plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    return()


def plot_base_probability_vs_uplift(data_probability,
                                    data_uplift,
                                    file_name='base_prob_vs_uplift.png',
                                    k=100000):
    """This function plots the predicted conversion probability vs. the
    predicted uplift. Note that the probabilities are not exactly "centered."
    This function was mostly for testing purposes.
    This could in principle show if there is some group that can be identified
    with response modeling (i.e. only predicting conversion probability, not
    uplift!) that is better or worse affected than the average user. This
    would enable you to do uplift modeling for that group but using a simpler
    approach with only response modeling.

    Args:
    data_probability (np.array([float, ...])): Vector of predicted conversion
     probabilities.
    file_name (str): Name of file where the plot should be stored.
    k (int): Size of the sliding window to smooth the uplift predictions.
    """
    # 'k' sets the size of the sliding window.
    idx = np.argsort(data_probability)  # Sort ascending
    data_probability = data_probability[idx]
    # Create sliding average
    probability_tmp = np.array([np.mean(data_probability[i:(i + k)])
                                for i in range(len(data_probability) - k)])
    data_uplift = data_uplift[idx]
    uplift_tmp = np.array([np.mean(data_uplift[i:(i + k)])
                           for i in range(len(data_uplift) - k)])
    plt.plot(probability_tmp, uplift_tmp)
    plt.xlabel('p(y|do(t=0))')
    plt.ylabel('p(y|x, do(t=1)) - p(y|x, do(t=0))')
    plt.title('Predicted conversion probability vs. uplift')
    plt.savefig(file_name)
    plt.close()
    return


def plot_uplift_vs_base_probability(data_probability,
                                    data_uplift,
                                    file_name='uplift_vs_base_prob.png',
                                    k=100000):
    """Function for plotting uplift vs. conversion probability. If this
    plot even vaguely resemples a 'V', then it indicates that there are
    samples where the uplift could not be simplified and derived from a
    response model. In technical terms, it implies that the uplift is not
    an injective function of the conversion probability. This is mostly
    a sanity check.

    Args:
    data_probability (np.array([float, ...])): Vector with predicted
    conversion probabilities for all samples in vector.
    data_uplift (np.array([float, ...])): Vector with predicted uplift
    probabilities for all samples.
    file_name (str): Name of file where plot is to be stored.
    k (int): Size of sliding window for smoothing of uplift

    Notes: The sliding window is for uplift predictions. There is no reason
    to assume that they would behave particularly nicely w.r.t. the conversion
    probability, hence the sliding window is necessary to make the graph
    smooth.
    """
    # 'k' sets the size of the sliding window.
    idx = np.argsort(data_uplift)  # Sort ascending
    data_probability = data_probability[idx]
    # Create sliding average
    probability_tmp = np.array([np.mean(data_probability[i:(i + k)])
                                for i in range(len(data_probability) - k)])
    data_uplift = data_uplift[idx]
    uplift_tmp = np.array([np.mean(data_uplift[i:(i + k)])
                           for i in range(len(data_uplift) - k)])
    plt.plot(uplift_tmp, probability_tmp)
    plt.ylabel('p(y|do(t=0))')
    plt.xlabel('p(y|x, do(t=1)) - p(y|x, do(t=0))')
    plt.title('Predicted uplift vs. conversion probability')
    plt.savefig(file_name)
    plt.close()
    return


