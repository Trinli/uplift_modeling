"""
Actual experiments for undersampling paper.
-The GP-models with approx 10k training observations might need to be split
into separate jobs.
-Not sure how large the trees end up being...

Misc:

# Hillstrom
# Train GP with 10k training samples. Perhaps show how larger training
# set with the tree-based approach can narrow the uncertainty further.
# Three example observations?'
# Also train Tree with the same dataset and the "full" training set.
# See if there are differences.

Criteo
Skip GP entirely?
High class-imbalance with the GP would cause disaster.
Does the model handling hich class-imbalance produce valid uncertainty
estimates? It should.
Show that it works. Perhaps compare to model without correction

Lenta
Dataset is so large that we only use thte tree-based model for this.
Experiments to show the beta-distribution for the uncertainty of the
response predictions, and then the combined beta-difference distribution
that illustrates how uncertainty in one propagates to the uplift
estimate.
-Estimate E(MSE) for max likelihood predictions (peak).
Note that this does not comprise a metric for refinement (Like AUC-ROC):
Maybe just estimate AUUC as refinement metric.
Store predictions for further processing? Basically we hope to somehow
quantify the width of the uncertainty predictions. Averag width, maybe.

Can I find the HIV-dataset?
Seems Prof. Tianxi Cai tcai@hsph@harvard.edu might know something
about that.
Despite all privacy concerns and trade secrets, I would imagine
there being at least one suitable dataset from a randomized
controlled trial publicly available.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import beta
import sys
import data.load_data as ld
import metrics.uplift_metrics as um
import models.honest_uplift_tree as honest_tree
import pickle
import models.gpy_dirichlet as GPY


def save_object(object, file_name='object.pickle'):
    """
    Function for saving object to disk. This is generic to the point
    where the object can be anything, including strings,
    lists, objects of other classes etc.

    Args:
    file_name (str): Name and location of file to write to.
    """
    with open(file_name, 'wb') as handle:
        pickle.dump(object, handle)

def load_object(file_name='tree_model.pickle'):
    """
    Functon for loading object from disk.

    Args:
    file_name (str): Name and location of file to read from.
    """
    with open(file_name, 'rb') as handle:
        object = pickle.load(handle)
    return object


def train_dirichlet_gp(data, file_name_stub="gp_tmp",
                       dataset=None, size=None, a_eps=0.1):
    """
    Args:
    data (data.load_data.DatasetCollection): Dataset to be used.
    file_name_stub (str): Used for storing results
    dataset (str): Used to store metrics
    size (int): Used to store metrics
    a_eps (float): Prior for gamma distributions.
    """
    # Import in function because this mess should not be imported unless
    # absolutely necessary (compatible with only a very specific version
    # of python, tensorflow, gpflow etc.).
    # Full training set for hillstrom has 10596 control observations and
    # 10710 treatment observations. Can use full data! :)
    X_t = data['gp_train', None, 'treatment']['X']
    y_t = data['gp_train', None, 'treatment']['y']
    X_c = data['gp_train', None, 'control']['X']
    y_c = data['gp_train', None, 'control']['y']

    import models.dirichlet_gp as GPD
    # Train Dirichlet Gaussian Process-model
    gp_model = GPD.DirichletUplift()
    gp_model.fit(X_t, y_t, X_c, y_c, a_eps=a_eps)  # Pass a_eps also?

    # Estimate metrics
    gp_pred = gp_model.predict_uplift(data['testing_set']['X'])
    gp_pred = gp_pred.astype(np.float32)
    gp_average_width = gp_model.mean_credible_interval_width(data['testing_set']['X'], 0.95)
    print("Average credible interval width for D-GP: {}".format(gp_average_width))
    gp_metrics = um.UpliftMetrics(data['testing_set']['y'], gp_pred, data['testing_set']['t'],
                                  test_description="Uncertainty with D-GP", algorithm="Dirichlet-GP",
                                  dataset=dataset + "_" + str(size),
                                  parameters=gp_average_width)  # Storing average width!
    gp_metrics.write_to_csv("./results/uncertainty_gp_results.csv")
    # Store model
    # save_object(gp_model, file_name_stub + "_gp_model.pickle")  # This GP cannot be serialized with pickle nor dill.

    print(gp_metrics)
    # Maybe test whether this metric can be reliably estimated with only 100 MC
    # observations per testing observation rather than 1000.

    # Visualizations
    # "Three" observations
    # First uncertainty of treatment response, then uncertainty of control response,
    # and lastly uncertainty of uplift. As both the tree and the gaussian process
    # are kind of double classifiers, the uncertainties for the response models
    # can easily be estimated.
    for i in range(10):
        test_item = data['testing_set']['X'][i, :].reshape((1, -1))
        # Histogram of uncertainty of prediction with treatment
        gp_params = gp_model.model_t.predict_params(test_item)
        mc_samples = gp_model.model_t.uncertainty_of_prediction(gp_params['fmu'],
                                                                gp_params['fs2'])
        # Distribution of uncertainty for prediction for observation if treated
        #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
        # HOW IS BANDWIDTH CHOSEN?
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(mc_samples[:, 1].reshape(-1, 1))
        X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        # Initialize plot
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[0, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=1}$")  # Width 0 to 0.5?
        ax[0].set_xlim([0, 0.5])
        #ax[0, 0].text(-3.5, 0.31, "p(y=1|x, t=1), Dirichlet GP")

        # Uncertainty of p(y=1|x, t=0) for D-GP:
        gp_params = gp_model.model_c.predict_params(test_item.reshape((1, -1)))
        mc_samples = gp_model.model_c.uncertainty_of_prediction(gp_params['fmu'],
                                                                gp_params['fs2'])
        # Distribution of uncertainty for prediction for observation if treated
        #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
        # HOW IS BANDWIDTH CHOSEN?
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(mc_samples[:, 1].reshape(-1, 1))
        log_dens = kde.score_samples(X_plot)
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=0}$")
        #ax[1, 0].text(-3.5, 0.31, "p(y=1|x, t=0), Dirichlet GP")

        # Uncertainty for D-GP
        mc_samples = gp_model.predict_uncertainty(test_item)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(np.array(mc_samples).reshape(-1, 1))
        X_plot = np.linspace(-1, 1, 2000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[2, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[1].plot(X_plot[:, 0], np.exp(log_dens), label="$u$")
        ax[1].set_xlim([-.5, .5])
        ax[1].axvline(0, color='black', linewidth=0.75)
        ax[0].legend()
        ax[1].legend()
        plt.savefig("./figures/" + file_name_stub + "_" + str(i) + '_gp_uncertainty.pdf')  # What format is required?
        #plt.savefig("./figures/gp.pdf")  # What format is required?
        # plt.show()
        plt.clf()

    # Uplift vs. credible interval width
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.clf()
    credible_intervals = gp_model.get_credible_intervals(data['testing_set']['X'])
    tmp = [item['width'] for item in credible_intervals]
    plt.scatter(gp_pred, tmp, alpha=0.2)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.savefig("./figures/" + file_name_stub + "_gp_scatter.pdf")
    #plt.savefig("./figures/gp_scatter.pdf")
    plt.clf()

    plt.hist(gp_pred, bins=100)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("#")
    plt.savefig("./figures/" + file_name_stub + str(i) + "gp_tau_histogram.pdf")
    plt.clf()

    return gp_metrics, gp_average_width


def dirichlet_gp_auuc_uncertainty(data, file_name_stub="gp_tmp",
                                  dataset=None, size=None,
                                  n_iterations=1000):
    # Import in function because this mess should not be imported unless
    # absolutely necessary (compatible with only a very specific version
    # of python, tensorflow, gpflow etc.).
    # Full training set for hillstrom has 10596 control observations and
    # 10710 treatment observations. Can use full data! :)
    X_t = data['gp_train', None, 'treatment']['X']
    y_t = data['gp_train', None, 'treatment']['y']
    X_c = data['gp_train', None, 'control']['X']
    y_c = data['gp_train', None, 'control']['y']

    import models.dirichlet_gp as GPD
    # Train Dirichlet Gaussian Process-model
    gp_model = GPD.DirichletUplift()
    gp_model.fit(X_t, y_t, X_c, y_c)

    # Estimate regular metrics:
    gp_pred = gp_model.predict_uplift(data['testing_set']['X'])
    gp_pred = gp_pred.astype(np.float32)
    gp_average_width = gp_model.mean_credible_interval_width(data['testing_set']['X'], 0.95)
    print("Average credible interval width for D-GP: {}".format(gp_average_width))
    gp_metrics = um.UpliftMetrics(data['testing_set']['y'], gp_pred, data['testing_set']['t'],
                                  test_description="Uncertainty with D-GP", algorithm="Dirichlet-GP",
                                  dataset=dataset + "_" + str(size),
                                  parameters=gp_average_width)  # Storing average width!
    gp_metrics.write_to_csv("./results/uncertainty_gp_results.csv")
    regular_auuc = gp_metrics.auuc

    # Estimate uncertainty of AUUC using random sampling
    auucs = []
    for i in range(n_iterations):
        gp_pred = gp_model.generate_sample(data['testing_set']['X'])
        gp_metrics = um.UpliftMetrics(data['testing_set']['y'], gp_pred, data['testing_set']['t'],
                                    test_description="Uncertainty in AUUC with D-GP", algorithm="Dirichlet-GP",
                                    dataset=dataset + "_" + str(size),
                                    parameters=gp_average_width)  # Storing average width!
        #gp_metrics.write_to_csv("./results/uncertainty_gp_results.csv")
        print(gp_metrics)
        gp_metrics.write_to_csv('./results/uncertainty_gp_' + file_name_stub + '.csv')
        auucs.append(gp_metrics.auuc)
    # Do something with auuc's:
    auuc_average = np.mean(auucs)
    auuc_std = np.std(auucs)
    print("AUUC-sampling average: {}".format(auuc_average))
    print("AUUC-sampling std: {}".format(auuc_std))
    plt.hist(auucs, bins=n_iterations//20)
    plt.xlabel("AUUC")
    plt.yticks([])
    plt.axvline(regular_auuc, color='red', linewidth=2)
    plt.savefig('figures/auuc_uncertainty_' + dataset + '.pdf')

    # Perhaps model the distribution. That is already being done with the GP.
    # At least add the AUUC for the "normal" metric.


def train_dirichlet_gpy(data, file_name_stub="gpy_tmp",
                        dataset=None, size=None, a_eps=0.1,
                        plots=False):
    """
    In contrast to the previous. this one uses the implementation based
    on Gpytorch!! Compatible with modern environments etc.

    Args:
    data (data.load_data.DatasetCollection): Dataset to be used.
    file_name_stub (str): Used for storing results
    dataset (str): Used to store metrics
    size (int): Used to store metrics
    a_eps (float): Prior for gamma distributions.
    plots (bool): If True, will produce plots.
    """
    # Import in function because this mess should not be imported unless
    # absolutely necessary (compatible with only a very specific version
    # of python, tensorflow, gpflow etc.).
    # Full training set for hillstrom has 10596 control observations and
    # 10710 treatment observations. Can use full data! :)
    print("This Dirichlet-GP is based on the GPytorch library!")
    X_t = data['gp_train', None, 'treatment']['X']
    y_t = data['gp_train', None, 'treatment']['y']
    X_c = data['gp_train', None, 'control']['X']
    y_c = data['gp_train', None, 'control']['y']

    #import models.dirichlet_gp as GPD
    # Train Dirichlet Gaussian Process-model
    gp_model = GPY.DirichletGPUplift()
    gp_model.fit(X_t, y_t, X_c, y_c, alpha_eps=a_eps)

    # Estimate metrics
    gp_pred = gp_model.predict_uplift(data['testing_set']['X'])
    gp_pred = gp_pred.astype(np.float32)
    gp_average_width = gp_model.mean_credible_interval_width(data['testing_set']['X'], 0.95)
    print("Average credible interval width for D-GPy: {}".format(gp_average_width))
    gp_metrics = um.UpliftMetrics(data['testing_set']['y'], gp_pred, data['testing_set']['t'],
                                  test_description="Uncertainty with D-GP", algorithm="Dirichlet-GP",
                                  dataset=dataset + "_" + str(size),
                                  parameters=gp_average_width)  # Storing average width!
    gp_metrics.write_to_csv("./results/uncertainty_gpy_results.csv")
    # Store model
    # save_object(gp_model, file_name_stub + "_gp_model.pickle")  # This GP cannot be serialized with pickle nor dill.

    print(gp_metrics)
    # Maybe test whether this metric can be reliably estimated with only 100 MC
    # observations per testing observation rather than 1000.

    # Visualizations
    # "Three" observations
    # First uncertainty of treatment response, then uncertainty of control response,
    # and lastly uncertainty of uplift. As both the tree and the gaussian process
    # are kind of double classifiers, the uncertainties for the response models
    # can easily be estimated.
    #idx = [np.argmax(gp_pred)]
    idx = [4016, 16028, 17061]
    print("Testing set observation with largest tau at {}".format(idx))
    # Print both predictions and uncertainty for this.

    if plots:
        # Plot 10 items _and_ a sample with large tau:
        for i in [j for j in range(10)] + idx:
            test_item = data['testing_set']['X'][i, :].reshape((1, -1))
            # Histogram of uncertainty of prediction with treatment
            #gp_params = gp_model.model_t.predict_params(test_item)
            # mc_samples = gp_model.model_t.uncertainty_of_prediction(gp_params['fmu'],
            #                                                         gp_params['fs2'])
            mc_samples = gp_model.model_t.generate_sample(test_item)
            # Distribution of uncertainty for prediction for observation if treated
            #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
            # HOW IS BANDWIDTH CHOSEN?
            kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(mc_samples[:, 1].reshape(-1, 1))
            X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
            log_dens = kde.score_samples(X_plot)
            # Initialize plot
            fig, ax = plt.subplots(1, 2, figsize=(8, 2))
            #fig.subplots_adjust(hspace=0.05, wspace=0.05)
            #ax[0, 0].fill(X_plot[:, 0], np.exp(log_dens))
            ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=1}$")  # Width 0 to 0.5?
            ax[0].set_xlim([0, 0.5])
            #ax[0, 0].text(-3.5, 0.31, "p(y=1|x, t=1), Dirichlet GP")

            # Uncertainty of p(y=1|x, t=0) for D-GP:
            # gp_params = gp_model.model_c.predict_params(test_item.reshape((1, -1)))
            # mc_samples = gp_model.model_c.uncertainty_of_prediction(gp_params['fmu'],
            #                                                         gp_params['fs2'])
            mc_samples = gp_model.model_c.generate_sample(test_item)
            # Distribution of uncertainty for prediction for observation if treated
            #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
            # HOW IS BANDWIDTH CHOSEN?
            kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(mc_samples[:, 1].reshape(-1, 1))
            log_dens = kde.score_samples(X_plot)
            #fig.subplots_adjust(hspace=0.05, wspace=0.05)
            #ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens))
            ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=0}$")
            #ax[1, 0].text(-3.5, 0.31, "p(y=1|x, t=0), Dirichlet GP")

            # Uncertainty for D-GP
            mc_samples = gp_model.generate_sample(test_item)
            kde = KernelDensity(kernel='gaussian', bandwidth=0.08).fit(np.array(mc_samples).reshape(-1, 1))
            X_plot = np.linspace(-1, 1, 2000)[:, np.newaxis]
            log_dens = kde.score_samples(X_plot)
            #fig.subplots_adjust(hspace=0.05, wspace=0.05)
            #ax[2, 0].fill(X_plot[:, 0], np.exp(log_dens))
            ax[1].plot(X_plot[:, 0], np.exp(log_dens), label="$u$")
            ax[1].set_xlim([-.5, .5])
            ax[1].axvline(0, color='black', linewidth=0.75)
            ax[0].legend()
            ax[1].legend()
            plt.savefig("./figures/" + file_name_stub + "_" + str(i) + '_gpy_uncertainty.pdf')  # What format is required?
            #plt.savefig("./figures/gp.pdf")  # What format is required?
            # plt.show()
            plt.clf()

    # Uplift vs. credible interval width
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.clf()
    credible_intervals = gp_model.get_credible_intervals(data['testing_set']['X'][:100000, :])

    tmp = [item['width'] for item in credible_intervals]
    plt.scatter(gp_pred[:10000], tmp[:10000], alpha=0.2)  # Plot 10k first observations.
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("95% credible interval width")
    plt.savefig("./figures/" + file_name_stub + "_gpy_scatter.pdf")
    #plt.savefig("./figures/gp_scatter.pdf")
    plt.clf()

    plt.hist(gp_pred, bins=100)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("#")
    plt.savefig("./figures/" + file_name_stub + "gpy_tau_histogram.pdf")
    plt.clf()

    return gp_metrics, gp_average_width


def train_gp(data, file_name_stub="gp_tmp", 
              dataset=None, size=None, a_eps=0.1):
    """
    Function for training uplift model with regular GP-classification
    models (rather than the Dirichlet-based GP that seems fishy).

    Args:
    data (data.load_data.DatasetCollection): Dataset to be used.
    file_name_stub (str): Used for storing results
    dataset (str): Used to store metrics
    size (int): Used to store metrics
    a_eps (float): Prior for gamma distributions.
    """
    # Full training set for hillstrom has 10596 control observations and
    # 10710 treatment observations. Can use full data! :)
    X_t = data['gp_train', None, 'treatment']['X']
    y_t = data['gp_train', None, 'treatment']['y']
    X_c = data['gp_train', None, 'control']['X']
    y_c = data['gp_train', None, 'control']['y']

    #import models.dirichlet_gp as GPD
    #sklearn.gaussian_process.GaussianProcessClassifier as GP  # Need to make an uplift model with these.
    import models.gp_uplift as GP
    # Train Dirichlet Gaussian Process-model
    gp_model = GP.GaussianProcessUplift()
    gp_model.fit(X_t, y_t, X_c, y_c)

    # Estimate metrics
    gp_pred = gp_model.predict_uplift(data['testing_set']['X'])
    #gp_pred = gp_pred.astype(np.float32)
    gp_average_width = gp_model.mean_credible_interval_width(data['testing_set']['X'], 0.95)
    print("Average credible interval width for D-GP: {}".format(gp_average_width))
    gp_metrics = um.UpliftMetrics(data['testing_set']['y'], gp_pred, data['testing_set']['t'],
                                  test_description="Uncertainty with D-GP", algorithm="Dirichlet-GP",
                                  dataset=dataset + "_" + str(size),
                                  parameters=gp_average_width)  # Storing average width!
    gp_metrics.write_to_csv("./results/uncertainty_gp_results.csv")
    # Store model
    # save_object(gp_model, file_name_stub + "_gp_model.pickle")  # This GP cannot be serialized with pickle nor dill.

    print(gp_metrics)
    # Maybe test whether this metric can be reliably estimated with only 100 MC
    # observations per testing observation rather than 1000.

    # Visualizations
    # "Three" observations
    # First uncertainty of treatment response, then uncertainty of control response,
    # and lastly uncertainty of uplift. As both the tree and the gaussian process
    # are kind of double classifiers, the uncertainties for the response models
    # can easily be estimated.
    for i in range(10):
        test_item = data['testing_set']['X'][i, :].reshape((1, -1))
        # Histogram of uncertainty of prediction with treatment
        gp_params = gp_model.model_t.predict_params(test_item)
        mc_samples = gp_model.model_t.uncertainty_of_prediction(gp_params['fmu'],
                                                                gp_params['fs2'])
        # Distribution of uncertainty for prediction for observation if treated
        #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
        # HOW IS BANDWIDTH CHOSEN?
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(mc_samples[:, 1].reshape(-1, 1))
        X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        # Initialize plot
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[0, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=1}$")  # Width 0 to 0.5?
        ax[0].set_xlim([0, 0.5])
        #ax[0, 0].text(-3.5, 0.31, "p(y=1|x, t=1), Dirichlet GP")

        # Uncertainty of p(y=1|x, t=0) for D-GP:
        gp_params = gp_model.model_c.predict_params(test_item.reshape((1, -1)))
        mc_samples = gp_model.model_c.uncertainty_of_prediction(gp_params['fmu'],
                                                                gp_params['fs2'])
        # Distribution of uncertainty for prediction for observation if treated
        #plt.hist(mc_samples[:, 1], bins=400, range=(0, 1))  # Column 1 contains estimates for the positive class.
        # HOW IS BANDWIDTH CHOSEN?
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(mc_samples[:, 1].reshape(-1, 1))
        log_dens = kde.score_samples(X_plot)
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=0}$")
        #ax[1, 0].text(-3.5, 0.31, "p(y=1|x, t=0), Dirichlet GP")

        # Uncertainty for D-GP
        mc_samples = gp_model.predict_uncertainty(test_item)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(np.array(mc_samples).reshape(-1, 1))
        X_plot = np.linspace(-1, 1, 2000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[2, 0].fill(X_plot[:, 0], np.exp(log_dens))
        ax[1].plot(X_plot[:, 0], np.exp(log_dens), label="$u$")
        ax[1].set_xlim([-.5, .5])
        ax[1].axvline(0, color='black', linewidth=0.75)
        ax[0].legend()
        ax[1].legend()
        plt.savefig("./figures/" + file_name_stub + "_" + str(i) + '_gp_uncertainty.pdf')  # What format is required?
        #plt.savefig("./figures/gp.pdf")  # What format is required?
        # plt.show()
        plt.clf()

    # Uplift vs. credible interval width
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.clf()
    credible_intervals = gp_model.get_credible_intervals(data['testing_set']['X'])
    tmp = [item['width'] for item in credible_intervals]
    plt.scatter(gp_pred, tmp, alpha=0.2)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.savefig("./figures/" + file_name_stub + "_gp_scatter.pdf")
    #plt.savefig("./figures/gp_scatter.pdf")
    plt.clf()

    plt.hist(gp_pred, bins=100)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("#")
    plt.savefig("./figures/" + file_name_stub + str(i) + "gp_tau_histogram.pdf")
    plt.clf()

    return gp_metrics, gp_average_width



def train_honest_tree(data, file_name_stub="tree_tmp",
                      dataset=None, size=None,
                      min_samples_leaf=None,
                      honest=True, max_leaf_nodes=None,
                      undersampling=False):
    # Train honest tree
    if min_samples_leaf is not None:
        tree_model = honest_tree.HonestUpliftTree(max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf)
    else:
        tree_model = honest_tree.HonestUpliftTree(max_leaf_nodes=max_leaf_nodes)
    if honest:
        if undersampling:
            print("Training honest tree with undersampling...")
            tree_model.honest_undersampling_fit(data)
            print("Optimal k: {}".format(tree_model.k))
        else:
            print("Training honest tree without undersampling...")
            tree_model.fit(data['tree_train']['X'], data['tree_train']['r'],
                        data['tree_val']['X'], data['tree_val']['y'],
                        data['tree_val']['t'])
    else:
        print("Training non-honest tree...")
        # How should the dataset size be taken into account here?
        tree_model.fit(data['tree_train']['X'], data['tree_train']['r'],
                       y=data['tree_train']['y'], t=data['tree_train']['t'])
    print("Done.")
    # Store model
    save_object(tree_model, file_name_stub + "_tree_model.pickle")

    # Estimate tree metrics:
    tree_pred = tree_model.predict_uplift(data['testing_set']['X'])
    tree_tau = np.array([item['tau'] for item in tree_pred])
    tree_average_width = tree_model.estimate_mean_credible_interval_width(
        data['testing_set']['X'])
    tree_metrics = um.UpliftMetrics(data['testing_set']['y'], tree_tau,
                                    data['testing_set']['t'],
                                    test_description="Uncertainty with Uplift Tree", algorithm="Honest Tree",
                                    dataset=file_name_stub,
                                    parameters=tree_average_width)  # Storing average width with results!
    tree_metrics.write_to_csv("./results/uncertainty_tree_results.csv")

    # Find observation with "large" uplift and plot that. Need to find a "large" through one single run, and then
    # use that same for other runs.
    # Find max in first 100k tree_pred
    idx = np.argmax(tree_tau[:100000])
    #idx = 19  # Found this by running line above with 100 000 observations on Criteo 2.
    print("Testing set observation with largest tau at {}".format(idx))
    # Print both predictions and uncertainty for this.

    # Plot 10 items _and_ a sample with large tau:
    for i in [j for j in range(10)] + [idx]:
        # Plot for Tree
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))
        test_item = data['testing_set']['X'][i, :].reshape((1, -1))
        tree_params = tree_model.predict_uplift(test_item)
        X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
        ax[0].plot(X_plot, beta.pdf(X_plot,
            tree_params[0]['alpha_t'], tree_params[0]['beta_t']),
                label="$p_{t=1}$")  # We want to change the scale here.
        ax[0].set_xlim([0, .5])
        #ax[0, 1].text(-3.5, 0.31, "p(y=1|x, t=1), Honest Tree")

        ax[0].plot(X_plot, beta.pdf(X_plot,
            tree_params[0]['alpha_c'], tree_params[0]['beta_c']), label="$p_{t=0}$")

        # Uncertainty of uplift: This one needs a different scale for the x-axis!!
        tree_samples = tree_model.generate_sample(test_item)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(tree_samples.reshape(-1, 1))
        X_plot = np.linspace(-1, 1, 2000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        ax[1].plot(X_plot[:, 0], np.exp(log_dens), label="$u$")
        ax[1].set_xlim([-.5, .5])
        # ax[1].spines['left'].set_position('zero')  # This was not it. Scale to the left, but line through origo.
        ax[1].axvline(0, color='black', linewidth=0.75)

        # Maybe change visualization so that one image contains all three plots (two as lines).
        #plt.title("Uncertainty of responses and uplift")  # Title not needed for paper
        #plt.legend()
        ax[0].legend()
        ax[1].legend()
        #plt.figure(figsize=(5, 10))
        plt.savefig("./figures/" + file_name_stub + "_" + str(i) + '_tree_uncertainty.pdf')  # What format is required?
        # plt.show()
        plt.clf()
        plt.hist(tree_samples)
        plt.savefig("./figures/tmp/" + file_name_stub + "_" + str(i) + '_tree_uncertainty_histogram.pdf')  # What format is required?
        plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.clf()
    # Scatter plot for predictions vs. width of credible intervals
    tree_width = np.array([item['hpd']['width'] for item in tree_pred])
    plt.scatter(tree_tau[:10000], tree_width[:10000], alpha=0.5)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("95% credible interval width")
    plt.savefig("./figures/" + file_name_stub + "_tree_scatter.pdf")
    plt.clf()
    plt.hist(tree_tau, bins=100)
    plt.xlabel(r"$\hat{\tau}(x)$")
    plt.ylabel("#")
    plt.savefig("./figures/" + file_name_stub + "_tree_tau_histogram.pdf")
    plt.clf()
    print("Average width of 95% credible interval: {}".format(tree_average_width))
    print(tree_metrics)
    #return tree_model  # Quick hack to extract model.
    return tree_metrics, tree_average_width


def find_tree_parameters():
    """
    Function that uses method above and cross-validation (?) to find
    best parameters.
    """
    # Load data. Starbucks.
    data_format = ld.STARBUCKS_FORMAT
    data_format['file_name'] = './datasets/' + data_format['file_name']
    data = ld.DatasetCollection(data_format['file_name'], data_format=data_format)
    # Get a 32k training set (actually 16k split 1/2 tree structure, 1/2 calibration)
    size = 32000
    data.add_set('tree_train', 0, int(size/2))
    data.add_set('tree_val', int(size/2), size)

    min_sample_leaf_list = [2**i for i in range(4, 13)]  # From 64 to 2**12=4096, experiments were run with 100
    max_leaf_nodes_list = [2**i for i in range(1, 9)]  # 81 for Criteo, 34 for Hillstrom, 12 for Starbucks in main experiments
    results = []
    for min_sample_leaf in min_sample_leaf_list:
        res_over_leafs = []
        for max_leaf_nodes in max_leaf_nodes_list:
            tmp = train_honest_tree(data=data,
                min_samples_leaf=min_sample_leaf,
                max_leaf_nodes=max_leaf_nodes) #, honest=False)
            res_over_leafs.append((tmp, min_sample_leaf, max_leaf_nodes))
        results.append(res_over_leafs)
    for res in results:
        tmp = [item[0][1] for item in res]
        print(tmp)
    # Larger min_sample_leaf leads to narrower average CI - as expected
    # min_sample_leaf 4098 implies splitting training data in two leafs...
    # However, AUUC is not particularly good
    # Max_leaf_nodes could be smaller for optimal AUUC?
    tmp_ = []
    for res in results:
        tmp = [item[0][0].auuc for item in res]
        print(tmp)
        tmp_.append(tmp)


    # Drop last row
    tmp = [res for res in results]  #[:-1]]
    # Drop two last columns:
    tmp = [res[:-2] for res in tmp]
    aci = []
    for line in tmp:
        tmp_ = [item[0][1] for item in line]
        print(tmp_)
        aci.append(tmp_)
    aci = np.array(aci)

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    fig.tight_layout(pad=4)
    im_aci = ax[0].imshow(aci, cmap='plasma')
    plt.colorbar(im_aci, ax=ax[0], fraction=0.05)
    # Add ticks and something.
    #plt.yticks([i for i in range(8)], [str(item) for item in min_sample_leaf_list[:-1]])
    ax[0].set_yticks([i for i in range(8)], [str(item) for item in min_sample_leaf_list[:-1]])
    #plt.xticks([i for i in range(6)], [str(item) for item in max_leaf_nodes_list[:-2]])
    ax[0].set_xticks([i for i in range(6)], [str(item) for item in max_leaf_nodes_list[:-2]])
    ax[0].set_title("Average CI")
    #plt.ylabel('Min. samples in node')
    #plt.xlabel('Max. number of leaf nodes')
    ax[0].set_ylabel('Min. samples in node')
    ax[0].set_xlabel('Max. number of leaf nodes')

    auuc = []
    for line in tmp:
        tmp_ = [item[0][0].auuc for item in line]
        auuc.append(tmp_)
    auuc = np.array(auuc)
    im_auuc = ax[1].imshow(auuc, cmap='viridis')
    plt.colorbar(im_auuc, ax=ax[1], fraction=0.05)
    #plt.yticks([i for i in range(8)], [str(item) for item in min_sample_leaf_list[:-1]])
    #plt.xticks([i for i in range(6)], [str(item) for item in max_leaf_nodes_list[:-2]])
    ax[1].set_yticks([i for i in range(9)], [str(item) for item in min_sample_leaf_list])
    ax[1].set_xticks([i for i in range(8)], [str(item) for item in max_leaf_nodes_list[:-2]] + [r'$2^7$', r'$2^8$'])
    ax[1].set_title('AUUC')
    #plt.ylabel('Min. samples in node')
    plt.xlabel('Max. number of leaf nodes')
    #ax[1].set_xlabel('Max. number of leaf nodes')
    plt.savefig('grid_search_aci_tree_64k.pdf')
    #plt.show()

def parse_file(result_file='./results/uncertainty/uncertainty_tree_results.csv'):
    """
    """
    import re
    import csv
    rows = []
    with open(result_file, 'r') as handle:
        reader = csv.reader(handle, delimiter=';')
        headers = reader.__next__()
        for row in reader:
            rows.append(row)
    #print("Dataset \tModel \tSize \tHonest \tUndersampling \tAUUC \t Average CI \tEUCE \tMUCE \tE(MSE)")
    values = []
    for row in rows:
        # append all to array and sort
        tmp_array = []
        tmp = re.split('_', row[1])
        tmp_array.append(tmp[0])  # Dataset
        tmp_array.append(tmp[1])  # Size
        tmp_array.append(row[3])  # Model
        try:
            tmp_array.append(tmp[3] == 'True')
        except:
            tmp_array.append(None)
        try:
            tmp_array.append(tmp[4] == 'True')
        except:
            tmp_array.append(None)
        tmp_array.append(row[7])  # AUUC
        tmp_array.append(row[4])  # Average CI
        tmp_array.append(row[10])  # EUCE
        tmp_array.append(row[11])  # MUCE
        tmp_array.append(row[19])  # E(MSE)
        tmp_tuple = (tmp_array[0], int(tmp_array[1]), tmp_array[2],
                     bool(tmp_array[3]), bool(tmp_array[4]),
                     float(tmp_array[5]), float(tmp_array[6]),
                     float(tmp_array[7]), float(tmp_array[8]),
                     float(tmp_array[9]))
        values.append(tmp_tuple)
    dtype = [('dataset', 'S12'), ('size', int), ('model', 'S11'), ('honest', bool),
             ('undersampling', bool), ('auuc', float), ('average_ci', float),
             ('euce', float), ('muce', float), ('emse', float)]
    output = np.array(values, dtype=dtype)
    output = np.sort(output, order=['dataset', 'model', 'undersampling', 'size'])
    return output


def parse_result_file(result_file='./results/uncertainty/uncertainty_tree_results.csv'):
    """
    Function for parsing results from csv-file and printing them in
    latex-format.

    Args:
    result_file (str): CSV-file containing results as writteh to file by
     DatasetCollection.write_to_csv().
    """
    # Print LATEX, latex table code:
    output = parse_file(result_file)
    latex_code = """
\\begin{table}[t]
"""
    latex_code += "\caption{"
    latex_code += "{}".format("Some title")
    latex_code += "}"
    latex_code += "\n"
    latex_code += """\label{sample-table}
\\begin{center}
\\begin{tabular}{llllllll}
\multicolumn{1}{c}{\\bf PART}  &\multicolumn{1}{c}{\\bf DESCRIPTION}
\\\\ \hline \\\\
"""
    # Column names
    #print("Dataset \tModel \tSize \tUndersampling \tAUUC \t Average CI \tEUCE \tMUCE \tE(MSE)")
    latex_code += "Dataset &Size &Model &Undersampling &mAUUC &Average CI &EUCE & $\mathbf{E}(MSE)$ \\\\ \n"
    for row in output:
        latex_code += """{} &{} &{} &{} &{:.5f} &{:.5f} &{:.5f} &{:.5f} \\\\""".format(
            row['dataset'].astype(str), row['size'], row['model'].astype(str), row['undersampling'],
            1000 * row['auuc'], row['average_ci'], row['euce'], row['emse']
        )
        latex_code +="\n"
    latex_code += """\end{tabular}
\end{center}
\end{table}
"""
    print(latex_code)


def plots():
    # Plot
    # Pick rows from output where
    # Criteo 2 with undersampling
    # Hillstrom and Starbucks without undersampling
    # -skip rows corresponding to 500 observations
    # -same plot for Criteo and the others? X-axis do not overlap.
    # -separate plot for DGP?
    result_file='./results/uncertainty/uncertainty_tree_results.csv'
    output = parse_file(result_file)
    tmp = [row for row in output if ((row['dataset'] == b'criteo2' and bool(row['undersampling']) is True) 
        or (row['dataset'] in [b'hillstrom', b'starbucks'] and bool(row['undersampling']) is not True))
        and row['size'] != 500]
    criteo_aci = [item[6] for item in tmp[:7]]
    criteo_size = [item[1] for item in tmp[:7]]
    criteo_auuc = [item[5] for item in tmp[:7]]
    hillstrom_aci = [item[6] for item in tmp[7:13]]
    hillstrom_size = [item[1] for item in tmp[7:13]]
    hillstrom_auuc = [item[5] for item in tmp[7:13]]
    starbucks_aci = [item[6] for item in tmp[13:]]
    starbucks_size = [item[1] for item in tmp[13:]]
    starbucks_auuc = [item[5] for item in tmp[13:]]
    result_file_2 = './results/uncertainty/uncertainty_gpy_results.csv'  # now GPY-results!
    output = parse_file(result_file_2)
    tmp = [row for row in output if row['dataset'] in [b'hillstrom', b'starbucks'] and row['size'] != 500]
    hillstrom_aci_gp = [item[6] for item in tmp[:6]]
    hillstrom_size_gp = [item[1] for item in tmp[:6]]
    hillstrom_auuc_gp = [item[5] for item in tmp[:6]]
    starbucks_aci_gp = [item[6] for item in tmp[6:]]
    starbucks_size_gp = [item[1] for item in tmp[6:]]
    starbucks_auuc_gp = [item[5] for item in tmp[6:]]
    # DGP models:
    plt.plot([i + 1 for i, _ in enumerate(hillstrom_aci_gp)], hillstrom_aci_gp, label='Hillstrom DGP', linestyle='dashed', color='tab:blue')
    plt.plot([i + 1 for i, _ in enumerate(starbucks_aci_gp)], starbucks_aci_gp, label='Starbucks DGP', linestyle='solid', color='tab:blue')
    plt.scatter([i + 1 for i, _ in enumerate(hillstrom_aci_gp)], hillstrom_aci_gp, color='tab:blue')
    plt.scatter([i + 1 for i, _ in enumerate(starbucks_aci_gp)], starbucks_aci_gp, color='tab:blue')
    # Tree models:
    plt.plot([i + 1 for i, _ in enumerate(hillstrom_aci)], hillstrom_aci, label='Hillstrom Tree', linestyle='dashed', color='tab:orange')
    plt.plot([i + 1 for i, _ in enumerate(starbucks_aci)], starbucks_aci, label='Starbucks Tree', linestyle='solid', color='tab:orange')
    plt.scatter([i + 1 for i, _ in enumerate(hillstrom_aci)], hillstrom_aci, color='tab:orange')
    plt.scatter([i + 1 for i, _ in enumerate(starbucks_aci)], starbucks_aci, color='tab:orange')
    plt.xticks([i + 1 for i, _ in enumerate(starbucks_aci)], [str(item) for item in starbucks_size])
    #plt.xscale('log')
    plt.ylabel('Average CI (95%)')
    plt.xlabel('Training set size')
    plt.legend()
    plt.savefig('average_ci_tree_gpy.pdf')


def plot_a_eps():
    """
    Code to plot AUUC, Average CI and Loss over different alpha_eps for gpy on Starbucks 32k.
    results needs to contain the following.
    """
    # Results with mean negative log-likelihood loss for _transformed_ _outcome_, not actual outcome (label). 
    # Log-likelihoods not comparable.
    # results = []
    # results.append({'auuc': 0.00263421669, 'a_eps': 0.001, 'aci': 0.00027660472551360726, 'loss_c': 2.98, 'loss_t': 3.207, 'loss': 6.186999999999999})
    # results.append({'auuc': 0.00265008551, 'a_eps': 0.01, 'aci': 0.005799875129014254, 'loss_c': 2.66, 'loss_t': 2.842, 'loss': 5.502000000000001})
    # results.append({'auuc': 0.00267331973, 'a_eps': 0.1, 'aci': 0.054398782551288605, 'loss_c': 2.162, 'loss_t': 2.261, 'loss': 4.423})
    # results.append({'auuc': 0.00225373504, 'a_eps': 1.0, 'aci': 0.06630009412765503, 'loss_c': 1.235, 'loss_t': 1.261, 'loss': 2.496})
    # #results.append({'auuc': 0.00262101812, 'a_eps': 2.0, 'aci': 0.05559254437685013, 'loss_c': 0.796, 'loss_t': 0.797, 'loss': 1.593})
    # #results.append({'auuc': 0.00228954741, 'a_eps': 4.0, 'aci': 0.04373819753527641, 'loss_c': 0.272, 'loss_t': 0.267, 'loss': 0.539})
    # results.append({'auuc': 0.00203525177, 'a_eps': 10.0, 'aci': 0.034580718725919724, 'loss_c': -0.511, 'loss_t': -0.524, 'loss': -1.035})
    # results.append({'auuc': -7.05306351e-05, 'a_eps': 100.0, 'aci': 0.0610869862139225, 'loss_c': -2.631, 'loss_t': -2.622, 'loss': -5.253})
    # Results with MNLL for original outcome (label). Results comparable
    results = []
    #results.append({'auuc': 0.00260295353,'mnll': 0.11296482443415755, 'a_eps': 0.001, 'aci': 0.0002701639896258712})
    results.append({'auuc': 0.00248798802, 'mnll': 0.08887322752811087, 'a_eps': 0.00390625, 'aci': 0.0017925185384228826})
    results.append({'auuc': 0.00250783746, 'mnll': 0.07775140859063685, 'a_eps': 0.0078125, 'aci': 0.0044718957506120205})
    #results.append({'auuc': 0.00271431102, 'mnll': 0.0740577307871572, 'a_eps': 0.01, 'aci': 0.006069921888411045})
    results.append({'auuc': 0.0024476778, 'mnll': 0.06831228948384524, 'a_eps': 0.015625, 'aci': 0.010984279215335846})
    results.append({'auuc': 0.00240022503, 'mnll': 0.06305571996883373, 'a_eps': 0.03125, 'aci': 0.025347460061311722})
    results.append({'auuc': 0.0025829298, 'mnll': 0.06722003487026086, 'a_eps': 0.0625, 'aci': 0.04673637077212334})
    #results.append({'auuc': 0.00267722593, 'mnll': 0.08058403760252986, 'a_eps': 0.1, 'aci': 0.05682007595896721})
    results.append({'auuc': 0.00254280552, 'mnll': 0.09070599398878403, 'a_eps': 0.125, 'aci': 0.057021159678697586})
    results.append({'auuc': 0.00254235032, 'mnll': 0.1469860236516688, 'a_eps': 0.25, 'aci': 0.04154525324702263})
    results.append({'auuc': 0.00235925936, 'mnll': 0.24735786883253605, 'a_eps': 0.5, 'aci': 0.058674633502960205})
    results.append({'auuc': 0.00219242583, 'mnll': 0.37705111638270317, 'a_eps': 1.0, 'aci': 0.06203700974583626})
    #results.append({'auuc': , 'mnll': , 'a_eps': 2.0, 'aci': })
    #results.append({'auuc': , 'mnll': , 'a_eps': 4.0, 'aci': })
    #results.append({'auuc': 0.00195265899, 'mnll': 0.6468956182897091, 'a_eps': 10.0, 'aci': 0.03545562922954559})
    #results.append({'auuc': -0.00112227568, 'mnll': 0.6968150579985232, 'a_eps': 100.0, 'aci': 0.06871015578508377})
    auuc = [item['auuc'] * 1000 for item in results]  # Changing unit to milli-AUUC
    aci = [item['aci'] for item in results]
    loss = [item['mnll'] for item in results]
    a_eps = [str(item['a_eps']) for item in results]

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)

    x_ticks = [r'$2^{-8}$', r'$2^{-7}$', r'$2^{-6}$', r'$2^{-5}$', r'$2^{-4}$', r'$2^{-3}$', r'$2^{-2}$', r'$2^{-1}$', r'$2^{0}$']
    #ax[0].set_ylabel('Min. samples in node')
    ax[0].plot(auuc, label='mAUUC')
    #ax[0].set_xticks([i for i, _ in enumerate(auuc)], a_eps)
    ax[0].set_xticks([i for i, _ in enumerate(auuc)], x_ticks)
    ax[0].set_ylabel('mAUUC')
    #ax[0].legend()
    #ax[0].set_title('AUUC')

    ax[1].plot(aci, label='Average CI')
    #ax[1].set_xticks([i for i, _ in enumerate(aci)], a_eps)
    ax[1].set_xticks([i for i, _ in enumerate(aci)], x_ticks)
    ax[1].set_ylabel('Average CI')
    #ax[1].legend()

    #ax2 = ax.twinx()
    ax[2].plot(loss, label='Mean negative log-likelihood')
    #ax[2].set_xticks([i for i, _ in enumerate(loss)], a_eps)
    ax[2].set_xticks([i for i, _ in enumerate(loss)], x_ticks)
    ax[2].set_ylabel('MNLL')
    #ax[2].legend()
    ax[2].set_xlabel(r"$\alpha_{\epsilon}$")
    plt.savefig('./figures/alpha_eps_starbucks_32k.pdf')
    #plt.show()
    plt.clf()


if __name__ == "__main__":
    # 0. Collect some args and run program accordingly.
    print("Use as 'python -m experiments.uncertainty_experiments model dataset training_set_size max_leaf_nodes honest undersampling")
    print("OR 'python -m experiments.uncertainty_experiments model dataset training_set_size alpha_epx")
    print("OR 'python -m experiments.uncertainty_experiments metrics result_file.csv'")
    parameters = sys.argv
    # 1. Load appropriate dataset
    tmp = parameters[1]
    if tmp == 'metrics':
        # Read result file and print in latex format
        # Exit program
        sys.exit(0)
    model = parameters[1]
    dataset = parameters[2]
    if dataset == 'criteo1':
        data_format = ld.CRITEO_FORMAT
        data_format['file_name'] = './datasets/criteo1/criteo1_1to1.csv'
    elif dataset == 'criteo2':
        data_format = ld.CRITEO_FORMAT
        #data_format['file_name'] = './datasets/criteo2/criteo2_1to1.csv'  # Replace with original?
        data_format['file_name'] = './datasets/criteo2/criteo-research-uplift-v2.1.csv'
    elif dataset == 'hillstrom':
        data_format = ld.HILLSTROM_FORMAT_1
        data_format['file_name'] = './datasets/' + data_format['file_name']
    elif dataset == 'voter':
        data_format = ld.VOTER_FORMAT
        data_format['file_name'] = './datasets/voter/voter_1_to_1_.5_dataset.csv'
    elif dataset == 'zenodo':
        data_format = ld.ZENODO_FORMAT
        data_format['file_name'] = './datasets/zenodo_modified.csv'
    elif dataset == 'lenta':
        # Get Lenta-dataset
        data_format = ld.LENTA_FORMAT
        data_format['file_name'] = './datasets/' + data_format['file_name']
    elif dataset == 'starbucks':
        data_format = ld.STARBUCKS_FORMAT
        data_format['file_name'] = './datasets/' + data_format['file_name']
    elif dataset == 'starbucks_mini':
        data_format = ld.STARBUCKS_FORMAT
        data_format['file_name'] = './datasets/starbucks_mini.csv'
    data = ld.DatasetCollection(data_format['file_name'], data_format=data_format)

    # -- also reduce size as needed (e.g. 1k observations for GP test run)
    size = int(parameters[3])  # [1k, 5k, 10k, 25k, 50k, 100k, 250k, 500k, 1M]
    # Check that size is equal or below training + validation set size to
    # not use testing set for training.
    if model == 'tree':
        try:
            max_leaf_nodes = int(parameters[4])
        except:
            # If not available, not setting it.
            max_leaf_nodes = None

        try:
            honest = parameters[5]
            if honest == 'True':
                # Everything is read in as a string
                # Set to boolean true.
                honest = True
            else:
                honest = False
        except:
            honest = False  ## "False" as default?

        try:
            undersampling = parameters[6]
            if undersampling == 'True':
                # String when read from command line. Change to bool:
                undersampling = True
            else:
                undersampling = False
        except:
            undersampling = False
    elif model == 'gp' or model == 'gpy':
        try:
            a_eps = float(parameters[4])
        except:
            print("No a_eps provided. Setting a_eps=0.1.")
            a_eps = 0.1
        try:
            plot = bool(parameters[5])
        except:
            plot = False

    tmp_n = data['training_set']['X'].shape[0] + data['validation_set']['X'].shape[0]
    assert size <= tmp_n, "Cannot run experiment with training set size {} (not enough observations).".format(size)
    if model == 'tree':
        if honest:
            data.add_set('tree_train', 0, int(size/2))
            data.add_set('tree_val', int(size/2), size)
        else:
            data.add_set('tree_train', 0, size)
    elif model == 'gp' or model == 'gpy':
        data.add_set('gp_train', 0, size)

    # 2. Call appropriate function (model)
    # -- parameters, e.g. leaf size for tree, names for output files
    if model == 'tree':
        file_name_stub = dataset + "_" + str(size) + '_' + str(max_leaf_nodes) + '_' + str(honest) + '_' + str(undersampling)
    elif model == 'gp':
        file_name_stub = dataset + "_" + str(size) + '_' + 'a_eps_{}'.format(a_eps) # Should contain dataset and downsampling, maybe leaf size for tree    
    elif model == 'gpy':
        file_name_stub = dataset + "_" + str(size) + '_' + 'gpy_a_eps_{}'.format(a_eps) # Should contain dataset and downsampling, maybe leaf size for tree    
    # Or actually, write to same file, just pass appropriate parameters to the metrics-object!

    # 3. Pass on name stub for all files produced. Maybe also result file.
    if model == 'gp':
        tmp = train_dirichlet_gp(data, file_name_stub, dataset, size, a_eps=a_eps)
    elif model == 'gpy':
        tmp = train_dirichlet_gpy(data, file_name_stub, dataset, size, a_eps=a_eps, plots=plot)
    elif model == 'tree':
        # Makes no sense training with 1k observations if leaf size is 400.
        tmp = train_honest_tree(data, file_name_stub, dataset, size,
                                honest=honest, max_leaf_nodes=max_leaf_nodes,
                                undersampling=undersampling)  # 'honest' was being passed in position for min_sample_leaf
    elif model == 'auuc_uncertainty':
        tmp = dirichlet_gp_auuc_uncertainty(data, file_name_stub,
                                            dataset, size, n_iterations=2000)
    else:
        print("Model {} not specified. Select 'gp' or 'tree'.".format(model))

    print("Training done.")
    if model != 'auuc_uncertainty':
        print(tmp[0])  # Print metrics
