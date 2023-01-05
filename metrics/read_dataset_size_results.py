"""
Code for reading in dataset size experiment results.
"""
import csv
import matplotlib.pyplot as plt

path = './results/dataset_size/'


def read_file(tmp_file):
    """
    Function for reading in result files.
    """
    lines = []
    with open(path + tmp_file) as handle:
        file_reader = csv.reader(handle, delimiter=';')
        for line in file_reader:
            lines.append(line)
    return lines


def find_optimal_results(file_name_val):
    """
    Function for finding optimal validation set result from file
    and corresponding line in testing set results.

    Args:
    file_name_val (str): Filename and path of validation set results.
     Will add '_testing_set.csv' to the end to find testing set
     results.
    """
    # Read in file to find max. Find corresponding line in *_testing_set.csv.
    # Print model name, best k, and testing set metrics.
    lines = read_file(file_name_val)
    # Find max AUUC and corresponding k in validation set
    # AUUC at [7], improvement to rand at [8]
    # dataset at [1], k at [4]
    best_auuc = -10.0
    # Reading in training set to find best parameters
    for line in lines:
        try:
            tmp_auuc = float(line[7])
        except ValueError:
            # Skip header rows
            continue
        if best_auuc <= tmp_auuc:  # Taking the "last" good model as best if there is a tie.
            # Keep best k-values
            best_val_line = line
            best_k = line[4]
            best_auuc = tmp_auuc
    # Print the best validation set results
    # Test details, dataset, k, improvement over random, auuc
    print("=" * 40)
    print("Validation set results:")
    print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_val_line))
    print("Lines in validation set: {}".format(len(lines)))

    # Next focus on testing set results:
    testing_set_file = file_name_val + '_testing_set.csv'
    lines_test = read_file(testing_set_file)
    # Find line that matches dataset and best_k
    best_test_line = None
    for line in lines_test:
        # criteo_results has the same indexing as files. We should be picking the one line that matches the k defined in best_k above.
        if line[4] == best_k:  # If best_k from validation set equals the best k in the row of the testing set file, then store:
            # Store the best line. If there are multiple lines with the same k, the last one is used.
            if best_test_line != None:
                print("There are multiple rows with the same k in the testing set results. Using the last.")
            best_test_line = line
    print("Testing set results:")
    print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_test_line))
    print("Lines in testing set: {}".format(len(lines_test)))
    return best_test_line


if __name__ == '__main__':
    # Generate file names:
    datasets = ['criteo1', 'criteo2']
    undersampling_schemes = ['none', 'k_undersampling', 'split_undersampling']  # No split for cvt and nn
    models = ['dc_lr', 'cvt_lr', 'uplift_rf', 'uplift_neural_net']
    # List for model names
    configurations = []
    # Corrections come directly from undersampling scheme
    for dataset in datasets:
        for undersampling_scheme in undersampling_schemes:
            for model in models:
                # Skip invalid combinations:
                if model == 'cvt_lr' or model == 'uplift_neural_net':
                    if undersampling_scheme == 'split_undersampling':
                        continue
                # Set correction scheme:
                if undersampling_scheme == 'none':
                    correction = 'none'
                elif undersampling_scheme == 'k_undersampling':
                    correction = 'div_by_k'
                elif undersampling_scheme == 'split_undersampling':
                    correction = 'analytic'
                # Generate model name?
                tmp_txt = dataset + '_' + undersampling_scheme + '_'
                tmp_txt += model + '_' + correction
                tmp_dict = {'name': tmp_txt, 'dataset': dataset,
                            'undersampling_scheme': undersampling_scheme,
                            'model': model, 'correction': correction}
                configurations.append(tmp_dict)

    # Now generate file names for validation set and read in optimal results
    #rates = [1, .75, .5, .25, .15, .1, 0.05, 0.025, 0.0125]
    rates = [1, .5, .25, .1, 0.05, 0.025, 0.0125]
    # Iterate over rates and read optimal results for each configuration
    # Basically add list with all rates to every dict in configurations
    # where order matches rates. Fill in None if result not available.
    for i, configuration in enumerate(configurations):
        # Create some list to append optimal results to
        tmp_list = []
        for rate in rates:
            file_name = configuration['name'] + '_r' + str(rate) + '.csv'
            try:
                # Seems some of the neural nets did finish training and storing validation set
                # results, but did not have time to store testing set results. That
                # produces an error.
                try:
                    tmp = find_optimal_results(file_name)
                    try:
                        tmp_list.append(float(tmp[7]))  # AUUC stored in idx 7
                    except TypeError:
                        tmp_list.append(None)
                except TypeError:
                    tmp_list.append(None)
            except FileNotFoundError:
                # Files for all do not exist. E.g. neural nets have not successfully stored
                # results for all rates. 
                tmp_list.append(None)  # Add placeholder
        configurations[i]['results'] = tmp_list
    # Next, plot results with suitable details.
    # Arto was speculating about a Gaussian process with "error bars" for the visualization.
    # We don't really have a good estimate for variance and a Gaussian process would
    # implicitly assume that the variance is smaller when the grid size is smaller.
    # I.e. we have smaller gaps for smaller rates, but smaller dataset also results in
    # larger variance for the dataset, hence the variance should probably be higher
    # for the smaller rates than the larger ones, but the graphs will show the opposite.
    # Basically the problem is the kernel function. We also do not have a good estimate
    # for the variance of our point estimates.
    # -Try both and discard as needed?

    # One figure, all variants of one model? That might make sense. E.g. CVT-LR
    # with none, k-undersampling, and split-undersampling. That will produce
    # four figures with three curves each.
    # Present only criteo1 or criteo 2. Whichever looks better. Criteo2 would
    # be preferrable because it does not contain the "biases" Criteo was
    # complaining about. No "split-undersampling" for cvt and nn.
    # Or turn it around? Three undersampling strategies (none, k-undersampling
    # and split_undersampling) and four or two curves per image.
    # Maybe one large image with 10 curves? That would make the comparison clearest.
    # AUUC should in principle be constant for all models... in principle.

    # Print all configurations.
    # Models will need to be renamed better!
    # Criteo1_1to1 contains 7.8M observations,
    # Criteo2_1to1 contains 4M observations. 
    # Maybe exclude CVT-LR from results. It is just a line around 0 and makes all other results
    # harder to read.

    # Configs by base learner
    tmp_idx = dc_idx = [0, 4, 8]
    cvt_idx = [1, 5]
    rf_idx = [2, 6, 9]
    nn_idx = [3, 7]
    # Best of every model
    best_idx = [0, 5, 9, 7]
    # Combinations of these
    dc_cvt_idx = dc_idx + cvt_idx
    rf_nn_idx = rf_idx + nn_idx
    # By type of undersampling
    none_idx = [0, 1, 2, 3]
    k_idx = [4, 5, 6, 7]
    split_idx = [8, 9]
    # Only ones that seem to make any sense is eigther having all in one plot,
    # or then separate plots by model. But that does not really tell the story.
    # Alternatively we can just focus on the four "best" models (subjective opinion).
    # tmp_idx = [i for i in range(10)]
    tmp_idx = dc_idx + cvt_idx + rf_idx + nn_idx
    labels = ['DC-LR', 'CVT-LR', 'Uplift RF', 'Uplift NN',
              'DC-LR (strat. und.)', 'CVT-LR (strat. und.)',
              'Uplift RF (strat. und.)', 'Uplift NN (strat. und.)',
              'DC-LR (split und.)', 'Uplift RF (split und.)']
    # THis does not work. Having one color for a model and a line
    # type for an undersampling scheme. 9 curves in one figure is
    # just too much.
    linestyle = ['solid', 'solid', 'solid', 'solid',
                 'dashed', 'dashed', 'dashed', 'dashed',
                 'dashdot', 'dashdot']
    # Change rates to strings so that plotting will not adjust for the values.
    txt_rates = [str(item) for item in rates]

    # Define colors corresponding to models:
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color = [colors[0], colors[1], colors[2], colors[3],
             colors[0], colors[1], colors[2], colors[3],
             colors[0], colors[2]]

    def multiply(item, factor=1000):
        # Use to change AUUC to mAUUC.
        if item is None:
            return None
        else:
            return item * factor

    for i in tmp_idx:
        # One config at a t ime.
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        if i != 1:  # Drop CVT-LR
            plt.plot(txt_rates,
                    [multiply(item) for item in config['results']],
                    label=labels[i],
                    linestyle=linestyle[i],
                    color=color[i])

    # Move legend out of plot
    # ax.legend(loc='lower left', bbox_to_anchor=(1, .45))
    # plt.subplots_adjust(bottom=.1, left=.15)

    plt.legend(title="Model", ncol=2)
    #plt.title("AUUC over dataset size")  # No title for plots in paper
    plt.xlabel("Fraction of dataset")
    plt.ylabel("mAUUC")

    plt.savefig('criteo2_dataset_size.pdf')
    plt.show()

    plt.clf()

    #############################################
    # Trying out one main plot and two minor ones
    # to the right
    #############################################
    #############################################
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                             gridspec_kw={'width_ratios':[2, 1]})
    #axs['Left'].set_title('Plot on Left')
    axs['TopRight'].set_title('DC-LR')
    axs['BottomRight'].set_title('Uplift RF')

    for i in tmp_idx:
        # One config at a t ime.
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        if i != 1:  # Drop CVT-LR
            axs['Left'].plot(txt_rates,
                             [multiply(item) for item in config['results']],
                             label=labels[i],
                             linestyle=linestyle[i],
                             color=color[i])
    axs['Left'].legend(title="Model")  #, ncol=2)
    axs['Left'].label_outer()
    axs['Left'].set_xlabel("Fraction of dataset")
    axs['Left'].set_ylabel("mAUUC")
    #plt.xlabel("Fraction of dataset")
    #plt.ylabel("mAUUC")


    for i in dc_idx:
        # One config at a t ime.
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        if i != 1:  # Drop CVT-LR
            axs['TopRight'].plot(txt_rates,
                                 [multiply(item) for item in config['results']],
                                 label=labels[i],
                                 linestyle=linestyle[i],
                                 color=color[i])
    axs['TopRight'].label_outer()

    for i in rf_idx:
        # One config at a t ime.
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        if i != 1:  # Drop CVT-LR
            axs['BottomRight'].plot(txt_rates,
                                    [multiply(item) for item in config['results']],
                                    label=labels[i],
                                    linestyle=linestyle[i],
                                    color=color[i])
    axs['BottomRight'].label_outer()
    axs['BottomRight'].set_xticklabels([])

    plt.savefig("criteo2_dataset_size.pdf")

    # Move legend out of plot
    # ax.legend(loc='lower left', bbox_to_anchor=(1, .45))
    # plt.subplots_adjust(bottom=.1, left=.15)

    #plt.legend(title="Model", ncol=2)
    #plt.title("AUUC over dataset size")  # No title for plots in paper


    #############################################
    # Trying out four separate plots.
    #############################################
    #############################################
    # New plot (2x2) with one square for every base-learner
    txt_rates = ['1', '.5', '.25', '.1', '.05', '.025', '.0125']
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in dc_idx:
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        ax1.plot(txt_rates,
                 [multiply(item) for item in config['results']],
                 label=labels[i],
                 linestyle=linestyle[i],
                 color=color[i])
    ax1.set_title('DC-LR')
    ax1.legend(['none', 'strat. und.', 'split und.'])
    for i in cvt_idx:
        if i != 1:  # Drop CVT-LR
            config = configurations[i + 10]  # + 10 to get to criteo 2 results.
            ax2.plot(txt_rates,
                    [multiply(item) for item in config['results']],
                    label=labels[i],
                    linestyle=linestyle[i],
                    color=color[i])
    ax2.set_title('CVT-LR')
    ax2.legend(['strat. und.'])
    i = dc_idx[0]  # add DC-LR to all as reference
    config = configurations[i + 10]  # + 10 to get to criteo 2 results.
    ax2.plot(txt_rates,
             [multiply(item) for item in config['results']],
             label=labels[i],
             linestyle=linestyle[i],
             color=color[i])

    for i in rf_idx:
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        ax3.plot(txt_rates,
                 [multiply(item) for item in config['results']],
                 label=labels[i],
                 linestyle=linestyle[i],
                 color=color[i])
    ax3.set_title('Uplift RF')
    ax3.legend(['none', 'strat. und.', 'split und.'])
    i = dc_idx[0]  # add DC-LR to all as reference
    config = configurations[i + 10]  # + 10 to get to criteo 2 results.
    ax3.plot(txt_rates,
             [multiply(item) for item in config['results']],
             label=labels[i],
             linestyle=linestyle[i],
             color=color[i])

    for i in nn_idx:
        config = configurations[i + 10]  # + 10 to get to criteo 2 results.
        ax4.plot(txt_rates,
                 [multiply(item) for item in config['results']],
                 label=labels[i],
                 linestyle=linestyle[i],
                 color=color[i])
    ax4.set_title('Uplift NN')
    ax4.legend(['none', 'strat. und.', 'split und.'])
    i = dc_idx[0]  # add DC-LR to all as reference
    config = configurations[i + 10]  # + 10 to get to criteo 2 results.
    ax4.plot(txt_rates,
             [multiply(item) for item in config['results']],
             label=labels[i],
             linestyle=linestyle[i],
             color=color[i])


    for ax in fig.get_axes():
        ax.label_outer()
    #plt.legend(title="Model", ncol=2)
    plt.xlabel("Fraction of dataset")
    plt.ylabel("mAUUC")


    ax1.set(xlabel='x-label', ylabel='y-label')
    ax3.set(xlabel='x-label', ylabel='y-label')
    ax2.label_outer()
    ax4.label_outer()

    # Gaussian process stuff.
    if False:
        # Next the same with gaussian processes
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        kernel = 1 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        for config in configurations:
            # Fix following
            tmp_rates = np.array(rates).reshape(-1, 1)
            gaussian_process.fit(tmp_rates, config['results'])
            gaussian_process.kernel_
            mean_prediction, std_prediction = gaussian_process.predict(tmp_rates, return_std=True)
            # Crashes when value of results is None.
            plt.plot(tmp_rates, config['results'], label=r"Blahable", linestyle="dotted")
            plt.scatter(tmp_rates, config['results'], label="Observations")
            plt.plot(tmp_rates, mean_prediction, label="Mean prediction")
            plt.fill_between(
                tmp_rates.ravel(),
                mean_prediction - 1.96 * std_prediction,
                mean_prediction + 1.96 * std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
            )
            plt.legend()
            plt.xlabel("$x$")
            plt.ylabel("$f(x)$")
            _ = plt.title("Gaussian process regression on noise-free dataset")
            #plt.savefig("why_gps_dont_fit_this_problem.png")
            plt.show()
            # GP's do not seem reasonable at all in this context. Like at all.
            # Maybe we could fiddle around with parameters, but is there really
            # any reason to think that some parameters are more right than others?
