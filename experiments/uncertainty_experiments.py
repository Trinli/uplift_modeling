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
                       dataset=None, size=None):
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
        ax[0].plot(X_plot[:, 0], np.exp(log_dens), label="$p_{t=1}$")
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


def train_honest_tree(data, file_name_stub="tree_tmp",
                      dataset=None, size=None,
                      honest=True, max_leaf_nodes=None,
                      undersampling=False):
    # Train honest tree
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

    # Plot 10 items:
    for i in range(10):
        # Plot for Tree
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))
        test_item = data['testing_set']['X'][i, :].reshape((1, -1))
        tree_params = tree_model.predict_uplift(test_item)
        X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
        ax[0].plot(X_plot, beta.pdf(X_plot,
            tree_params[0]['alpha_t'], tree_params[0]['beta_t']),
                label="$p_{t=1}$")
        #ax[0, 1].text(-3.5, 0.31, "p(y=1|x, t=1), Honest Tree")

        ax[0].plot(X_plot, beta.pdf(X_plot,
            tree_params[0]['alpha_c'], tree_params[0]['beta_c']), label="$p_{t=0}$")

        # Uncertainty of uplift: This one needs a different scale for the x-axis!!
        tree_samples = tree_model.generate_sample(test_item)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(tree_samples.reshape(-1, 1))
        X_plot = np.linspace(-1, 1, 2000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        ax[1].plot(X_plot[:, 0], np.exp(log_dens), label="$u$")
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

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.clf()
    # Scatter plot for predictions vs. width of credible intervals
    tree_width = np.array([item['hpd']['width'] for item in tree_pred])
    plt.scatter(tree_tau, tree_width, alpha=0.5)
    plt.xlabel(r'$\hat{\tau}(x)$')
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
    return tree_metrics, tree_average_width


def parse_result_file(result_file='./results/uncertainty/uncertainty_tree_results.csv'):
    """
    Function for parsing results from csv-file and printing them in
    latex-format.

    Args:
    result_file (str): CSV-file containing results as writteh to file by
     DatasetCollection.write_to_csv().
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
    # Print LATEX, latex table code:
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


if __name__ == "__main__":
    # 0. Collect some args and run program accordingly.
    print("Use as 'python -m experiments.uncertainty_experiments model dataset training_set_size max_leaf_nodes honest undersampling")
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

    tmp_n = data['training_set']['X'].shape[0] + data['validation_set']['X'].shape[0]
    assert size <= tmp_n, "Cannot run experiment with training set size {} (not enough observations).".format(size)
    if honest:
        data.add_set('tree_train', 0, int(size/2))
        data.add_set('tree_val', int(size/2), size)
    else:
        data.add_set('tree_train', 0, size)
    data.add_set('gp_train', 0, size)

    # 2. Call appropriate function (model)
    # -- parameters, e.g. leaf size for tree, names for output files
    if model == 'tree':
        file_name_stub = dataset + "_" + str(size) + '_' + str(max_leaf_nodes) + '_' + str(honest) + '_' + str(undersampling)
    else:
        file_name_stub = dataset + "_" + str(size)  # Should contain dataset and downsampling, maybe leaf size for tree    
    # Or actually, write to same file, just pass appropriate parameters to the metrics-object!

    # 3. Pass on name stub for all files produced. Maybe also result file.
    if model == 'gp':
        tmp = train_dirichlet_gp(data, file_name_stub, dataset, size)
    elif model == 'tree':
        # Makes no sense training with 1k observations if leaf size is 400.
        tmp = train_honest_tree(data, file_name_stub, dataset, size,
                                honest, max_leaf_nodes=max_leaf_nodes,
                                undersampling=undersampling)
    elif model == 'auuc_uncertainty':
        tmp = dirichlet_gp_auuc_uncertainty(data, file_name_stub,
                                            dataset, size, n_iterations=2000)
    else:
        print("Model {} not specified. Select 'gp' or 'tree'.".format(model))

    print("Training done.")
    if model != 'auuc_uncertainty':
        print(tmp[0])  # Print metrics
