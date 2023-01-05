"""
Visualizations for effect of undersampling on uplift estimates.
Used in 2nd year presentation. Requires at least version 0.24.0
of scikit-learn for the isotonic regression plot.

Plot-producing functions:
plot_iso_uplift()
undersampling_for_classification()
plot_isotonic_function()
uplift_bar_plot()
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
import data.load_data as load_data
from sklearn.linear_model import LogisticRegression


# Parameters
PLOT_RANGE = [0, 100]  # Range of items plotted
UNDERSAMPLING_COEF = 0.996  # In [0, 1]
UPLIFT = 0.001  # E.g. 0.01 equals 1%
RESOLUTION = 10000  # Number of points estimated in [0, 1]
# Iso-uplift lines to plot (multiples of uplift):
ISO_UPLIFT_LINES = [2, 1, 0, -1]
UPLIFT_BASE_PROBABILITIES = [.001, .002, .003]
ZERO_BAR_SIZE = 0.1  # Make 0 in barchart non-zero to show up in plot.
# Number of datapoints generated for the isotonic plot:
ISOTONIC_DATAPOINTS = 50


def undersample(data, coef=UNDERSAMPLING_COEF):
    """
    Function returns new probability estimate after undersampling.

    Args:
    data (float or [float]): Number in [0, 1]. This is the baseline
    probability. After undersampling of the _negative_ class by coef
    (e.g. 0.9 implies that 90% of all negative samples are dropped).
    coef (float): Number in [0, 1].
    """
    if isinstance(data, list):
        tmp = [item / (item + (1 - item) * (1 - coef)) for item in data]
    elif isinstance(data, float):
        tmp = data / (data + (1 - data) * (1 - coef))
    else:
        raise ValueError
    return tmp


def plot_iso_line(tmp_label,
                  amount_of_uplift=UPLIFT,
                  plt=plt):
    """
    Auxiliary function for plot_iso_uplift(). Plots one curve.
    Function to plot one iso-uplift curve as function of baseline
    probability (p(y|do(t=0))). Can be called multiple times to
    get more iso-uplift curves in one plot.

    Args:
    tmp_label []: List to add labels to. Can be used to add separate
    labels to all curves in one plot.
    """
    # The control predictions stay constant for all treatment curves:
    c = [item * 1 / RESOLUTION for item in range(0, RESOLUTION)]
    c_new = undersample(c)
    # The treatment predictions are control predictions + amount_of_uplift
    t_tmp = [item + amount_of_uplift for item in c]
    t_new_tmp = undersample(t_tmp)
    diff_new_tmp = [item_t - item_c for item_c, item_t
                    in zip(c_new, t_new_tmp)]

    # Skip some items from plot? For negative uplift, the control probability
    # must be at least equally large (with opposite sign) to result in a valid
    # conversion probability:
    skip_items = sum([item + amount_of_uplift < 0 for item in c])
    plt.plot([item * 100 / RESOLUTION for item in
              range(max(skip_items, PLOT_RANGE[0]), PLOT_RANGE[1])],
             [item * 100 for item in
              diff_new_tmp[max(skip_items, PLOT_RANGE[0]):PLOT_RANGE[1]]])
    # Uncomment for labels
    #tmp_label.append('{}% uplift'.format(amount_of_uplift*100))
    return


def plot_iso_uplift(iso_uplift_lines=ISO_UPLIFT_LINES,
                    uplift=UPLIFT,
                    undersampling_coef=UNDERSAMPLING_COEF):
    """
    Function for plotting iso-uplift curves, i.e. uplift estimates
    after undersampling vs. baseline probability. The uplift estimate
    in undersampled data is dependent on both undersampling coefficient
    and p(y|do(t=0)) (baseline probability).
    """
    # List for labels:
    tmp_label = []
    # Plot desired iso-uplift lines (multiples of amount_of_uplift):
    #plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.15)
    #plt.gcf().subplots_adjust(left=0.15)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('xtick', labelsize=17*1.5-4)
    plt.rc('ytick', labelsize=17*1.5-4)
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    [plot_iso_line(tmp_label, item * uplift) for item in iso_uplift_lines]
    plt.grid(True)  # Add grid for easier visualization
    #plt.legend(tmp_label, fontsize=17*1.5)
    #plt.title("Iso-uplift lines vs. conversion probability \nwith" +
    #          " {}% undersampling ".format(undersampling_coef * 100) +
    #          "of negative samples")
    plt.xlabel(r'$p(y|do(t=0))$ [%]', fontsize=17*1.5)
    # The following is actually percentage points:
    #plt.ylabel('after undersampling [%]', fontsize=17*1.5)
    plt.ylabel(r'$\tau^{*}$ [%]', fontsize=17*1.5)
    plt.savefig("problem_with_undersampling.pdf")
    # Clear plot
    plt.clf()


def undersampling_for_classification(plt=plt):
    """
    This function plots how probabilities before undersampling
    vs. probabilities after undersampling. This is an idealization
    of what undersampling does to probabilities in classification.

    Args:
    plt (matplotlib.pyplot): Plot-object
    """
    p_y_normal = [i / RESOLUTION for i in range(PLOT_RANGE[0],
                                                PLOT_RANGE[1])]
    p_y_new = undersample(p_y_normal)

    plt.plot([item * 100 for item in p_y_normal],
             [item * 100 for item in p_y_new])
    plt.title("Effect of undersampling on probability " +
              "estimates \n({}%".format(UNDERSAMPLING_COEF * 100) +
              " of negative samples dropped)")
    plt.xlabel("p(y) without undersampling [%]")
    plt.ylabel("p(y) after undersampling [%]")
    plt.savefig("undersampling_for_classification.pdf")

    plt.clf()


def uplift_bar_plot(base_probabilities=UPLIFT_BASE_PROBABILITIES):
    """
    Function for plotting uplift ranges after undersampling vs.
    uplift estimates before undersampling. The ranges are
    defined as max and min uplift estimates with base probabilities
    defined in code.
    """
    def uplift_range(uplift,
                     base_probabilities=base_probabilities,
                     coef=UNDERSAMPLING_COEF):
        """
        Auxiliary function
        """
        tmp = []
        for item in base_probabilities:
            p_y_c = undersample(item, coef)
            p_y_t = undersample(item + uplift, coef)
            tmp.append(p_y_t - p_y_c)
        return tmp

    # List of uplifts before undersampling:
    x = [i * UPLIFT for i in ISO_UPLIFT_LINES[::-1]]
    # New ranges of uplift estimates as consequence of
    # undersampling AND base probability
    uplift_estimates = [uplift_range(item, base_probabilities)
                        for item in x]

    tmp_x = [item * 100 for item in x]
    tmp_height = [item[1] * 100 for item in uplift_estimates]
    ticks = ["{:.4}".format(item) for item in tmp_x]
    err = [[(item[1] - item[0]) * 100 for item in uplift_estimates],
           [(item[2] - item[1]) * 100 for item in uplift_estimates]]
    # Give probability estimate that equals zero a small value to make it
    # visible in plots:
    try:
        # Index not necessarily in list
        idx = tmp_height.index(0)
        tmp_height[idx] = ZERO_BAR_SIZE
    except ValueError:
        pass

    # Colors (use same colors from other plots):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_items = len(uplift_estimates)
    colors_tmp = colors[:n_items]
    colors_tmp = colors_tmp[::-1]  # Flip to match colors in iso-uplift plot
    plt.bar(tmp_x, tmp_height, width=.08, tick_label=ticks,
            yerr=err, color=colors_tmp)
    plt.xlabel("Uplift [%] (before undersampling)")
    plt.ylabel("Uplift [%] (after undersampling)")
    plt.title("Uplift before and after undersampling\n" +
              "{}% ".format(UNDERSAMPLING_COEF * 100) +
              "of negative samples dropped")
    plt.savefig("uplift_bar.png")

    plt.clf()


def plot_isotonic_function():
    """
    Function for plotting the piecewise constant function produced
    by isotonic regression (and some generated data).
    """
    data = np.random.randn(ISOTONIC_DATAPOINTS)
    # Class somewhat dependent on data:
    x_min = np.min(data)
    x_max = np.max(data)
    y = np.array([np.random.binomial(1, (tmp - x_min)/(x_max - x_min))
                  for tmp in data])
    model = IsotonicRegression()
    model.fit(X=data, y=y)
    plt.plot(model.X_thresholds_, model.y_thresholds_)
    plt.title("Isotonic regression")
    plt.xlabel("score")
    plt.ylabel("p(y|x)")
    plt.savefig("isotonic_regression.png")
    plt.clf()

def get_data():
    # 1. Load data
    format = load_data.DATA_FORMAT
    file_name = 'criteo_100k.csv'  
    #file_name = 'criteo-uplift.csv'
    path = './datasets/' + file_name
    # path = './datasets/criteo/' + file_name
    data = load_data.DatasetCollection(path, format)
    return data

def criteo_histograms(data=data):
    # 3. Train LR response model on control-data
    model = LogisticRegression()
    model.fit(data['training_set']['X'], data['training_set']['y'])
    # 4. Predict conversion rate for testing set based on model
    predictions = model.predict_proba(data['testing_set']['X'])

    # 5. Plot histogram
    #quantile_low = np.quantile(predictions[:, 1], q=0.025)
    quantile_low = 0  # Not actually quantile, just lower boundary for plot
    quantile_high = np.quantile(predictions[:, 1], q=0.975)
    #quantile_high = 0.003
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('xtick', labelsize=17*1.5-4)
    plt.rc('ytick', labelsize=17*1.5-4)
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.hist(predictions[:, 1]*100, bins=100, range=[0, quantile_high*100])
    plt.xlabel(r'$p(y|\bar{x}, do(t=0))$ [%]', fontsize=17*1.5)
    plt.ylabel('samples', fontsize=17*1.5)

    plt.savefig('criteo_p_y_histogram.png')
    plt.clf()

    # Another version with undersampling and calibration
    # Do we need undersampling for this, too? Probabilities seem super-low.
    k = 200
    training_data = data.k_undersampling(k=k, group_sampling='11')
    model.fit(training_data['X'], training_data['y'])
    new_predictions = model.predict_proba(data['validation_set']['X'])
    ir_model = IsotonicRegression(out_of_bounds='clip')
    ir_model.fit(new_predictions[:, 1], data['validation_set']['y'])
    testing_set_predictions = model.predict_proba(data['testing_set']['X'])
    calibrated_predictions = ir_model.predict(testing_set_predictions[:, 1])

    print("Average of corrected predictions: {}".format(np.mean(calibrated_predictions)))
    print("P(y) in training data: {}".format(np.mean(data['training_set']['y'])))
    #quantile_low = np.quantile(calibrated_predictions, q=0.025)
    quantile_low = 0
    #quantile_high = np.quantile(calibrated_predictions, q=0.9750)
    quantile_high = 0.0010
    plt.hist(calibrated_predictions,
             #, range=[quantile_low, quantile_high],
             density=True)  #, bins=100)
    plt.title('Distribution of predicted conversion rates')
    plt.xlabel('p(y|do(t=0))')
    #plt.show()
    plt.savefig('criteo_histogram_with_undersampling.png')
    plt.clf()


def plot_probability_difference():
    """
    Function for plotting a 1% difference in probabilities with
    different base probabilities. I.e. the first plot can have
    a 1% base rate, the second a 2% base rate. This over multiple
    s.
    """
    baserate_1 = 0.002
    baserate_2 = 0.02
    def undersampling(s, prob):
        return prob / (prob + s * (1 - prob))
    def k_undersampling(k, base_rate, probability):
        drop_prob = (1/k - base_rate) / (1 - base_rate)
        return undersampling(drop_prob, probability)

    probability_1 = [i * 0.001 for i in range(1, 99)]
    probability_2 = [i * 0.001 for i in range(2, 100)]
    probability_3 = [i * 0.001 + .0005 for i in range(2, 100)]
    new_prob_1 = [k_undersampling(10, baserate_1, tmp) for tmp in probability_1]
    new_prob_2 = [k_undersampling(10, baserate_1, tmp) for tmp in probability_2]
    new_prob_3 = [k_undersampling(10, baserate_1, tmp) for tmp in probability_3]
    diff = [item_2 - item_1 for item_1, item_2 in zip(new_prob_1, new_prob_2)]
    diff_2 = [item_3 - item_1 for item_1, item_3 in zip(new_prob_1, new_prob_3)]
    plt.plot(100 * np.array(probability_1), 100 * np.array(diff), label=r'$\tau(x) = 0.1\%$')
    plt.plot(100 * np.array(probability_1), 100 * np.array(diff_2), label=r'$\tau(x) = 0.15\%$')
    plt.xlabel(r"$p(y=1 | x, t=1), [\%]$")
    plt.ylabel(r"$\tau^*(x), [\%]$")
    plt.legend()
    plt.savefig("0.5percent_difference.pdf")

if __name__ == '__main__':
    # Run everything:
    data = get_data()
    plot_iso_uplift()
    undersampling_for_classification()
    uplift_bar_plot()
    plot_isotonic_function()
    criteo_histograms()
