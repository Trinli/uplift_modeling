"""
Implementation of an honest uplift tree with a twist...
This implementation will handle undersampling so that
the tree structure is inferred from undersampled data,
but the parameters for the leafs are estimated from
regular data. This way, the predictions of the leafs
will have statistical guarantees the same way that
Athey & al.'s honest tree does. The leaves will also
output the numbers used to estimate the samples explicitly
so that the uncertainties of the probabilities can be
quantified. The uncertainties should be evaluated at
training time to make prediction quicker (i.e. instead
of doing MC for every observation, it is done once for
every leaf).

Misc:
-Can this be extended to a forest?
-Maybe have the tree stop when the credible intervals
become too large? Like in RCIR.
"""
from copy import deepcopy
import itertools
import numpy as np
from scipy.stats import beta
from sklearn.tree import DecisionTreeRegressor
from metrics import uplift_metrics
from metrics import beta_difference  # Credible intervals given alpha and beta-parameters.



class HonestUpliftTree:
    """
    Class for honest tree following Athey & Imben's model that internally
    handles undersampling over given parameters, cross-validation to find
    the best model, and estimates Bayesian credible intervals in the leafs. 
    They used CART to train just a regular tree with "transformed outcome"
    (previously "revert label"). Then they estimate the leaf parameters
    using a separate training set.
    We will also be estimating leaf parameters from the honest set, but
    instead of using the transformed outcome (as we did to learn tree
    structure), we will be counting the number of positive and negative
    treated and untreated observations - all separately. Using this, we
    can then estimate the uncertainty of estimates and calculate e.g.
    HPD-intervals. The predict function could also return the alpha_1,
    alpha_0, beta_1, and beta_0 for the beta-difference distribution.

    First train a CART-model. Then evaluate the leaf parameters on an
    honest set. Estimate the validation set performance outside of this
    class (?). Or should undersampling be included here?
    Train with default parameters?
    """
    def __init__(self, max_leaf_nodes=None, min_samples_leaf=100,
                 auto_parameters=False):
        """
        Only initialize model. Training is accomplished with fit().

        In the honest step, make sure that all nodes are populated with a
        large enough sample.

        Args:
        min_sample_leaf (int): Minimum number of observations in a leaf.
        """
        self.tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                          min_samples_leaf=min_samples_leaf)
        self.node_dict = {}  # Placeholder for honest parameters
        self.k = 1  # Undersampling parameter if any
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.auto_parameters = auto_parameters

    def fit(self, X, r, honest_X=None, honest_y=None, honest_t=None, y=None, t=None):
        """
        X, y or data? Just feed the data-object? All undersampling etc for the
        training set already implemented. Can even split the training set in two
        to get separate honest set and validation set. Alternatively this could
        all be handled outside of this function.
        -Sounds like a better idea. Make one "base model" that is then wrapped
         in something that handles cross-validation etc.
        And how is the honest and undersampling blah blah estimated?
        """
        # 0. Split data into appropriate sets
        # 1. Train model
        if self.auto_parameters:
            # Find min_samples_leaf and max_leaf_nodes automatically from predefined list.
            # Search through a reasonable space
            max_leaf_nodes_list = [2**i for i in range(1, 8)]  # Up to 256 nodes
            min_samples_leaf_list = [2**i for i in range(6, 12)]  # Up to 2048 min
            parameter_list = itertools.product(max_leaf_nodes_list, min_samples_leaf_list)
            # Fit tree with five-fold cross-validation within the training set
            # based on auuc?
            best_auuc = -1.0  # Worst AUUC possible.
            cv_folds = 10
            n_samples = X.shape[0]
            fold_idx = [[i + j * n_samples // cv_folds for i in range(n_samples // cv_folds)] for j in range(cv_folds)]
            for max_leaf_nodes, min_samples_leaf in parameter_list:
                tmp_metrics = []  # Reset metrics for new parameters
                for i in range(cv_folds):
                    # N-fold cross-validation
                    tmp_fold_idx = [item for j, item in enumerate(fold_idx) if j != i]
                    # Unpack:
                    train_idx = [item for item in itertools.chain.from_iterable(tmp_fold_idx)]
                    cross_val_idx = fold_idx[i]
                    X_tmp = X[train_idx, :]
                    r_tmp = r[train_idx]
                    tmp_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                                    min_samples_leaf=min_samples_leaf)
                    tmp_model.fit(X_tmp, r_tmp)
                    # Evaluate model on hold-out set:
                    X_val = X[cross_val_idx, :]
                    y_val = y[cross_val_idx]
                    t_val = t[cross_val_idx]
                    tmp_pred = tmp_model.predict(X_val)
                    tmp_metric = uplift_metrics.UpliftMetrics(y_val, tmp_pred, t_val, k=2)
                    tmp_metrics.append(tmp_metric)
                # Average metrics over folds
                tmp_auuc = np.average([item.auuc for item in tmp_metrics])
                # 4. See if model is best so far
                if tmp_auuc > best_auuc:
                    best_auuc = tmp_auuc
                    best_max_leaf_nodes = max_leaf_nodes
                    best_min_samples_leaf = min_samples_leaf
                    #best_model = tmp_model
            self.tree = DecisionTreeRegressor(max_leaf_nodes=best_max_leaf_nodes,
                                              min_samples_leaf=best_min_samples_leaf)
            self.tree.fit(X, r)
            self.max_leaf_nodes = best_max_leaf_nodes
            self.min_samples_leaf = best_min_samples_leaf
        else:
            self.tree.fit(X, r)
        # 2. Predict what leafs honest observations fall into
        if honest_X is not None and honest_y is not None and honest_t is not None:
            self.honest_fit(honest_X, honest_y, honest_t)
        else:
            # Use training set! (Needs to be passed).
            print("Training non-honest model.")
            self.honest_fit(X, y, t)

    def honest_fit(self, honest_X, honest_y, honest_t):
        """
        Method for estimation of honest parameters. The tree needs to be trained first
        with the fit()-method.

        Notes:
        Validation set might not contain observations falling into all leafs,
        leaving some not honestly validated (and due to implementation, missing!).
        Is there a way to check that there is a node in the node_dict corresponding to
        all leaves. Perhaps throw and error? Honest estimation ends up not being
        particularly good if there are only few or no observations to estimate the
        honest parameters from.
        It is less of a problem if the training data is undersampled.

        """
        leaf_idx = self.tree.apply(honest_X)
        # 3. Estimate leaf parameters from the honest predictions
        # -Find sample idx's per node
        #tmp = [[] for i in range(self.tree.tree_.n_leaves)]  # We can create a list with elements for all _nodes_ and just leave some empty.
        #tmp = [[] for i in range(self.tree.tree_.node_count)]
        #[tmp[item].append(i) for i, item in enumerate(leaf_idx)]  # leaf_idx is where the honest samples are mapped
        tmp_n_t = [0 for i in range(self.tree.tree_.node_count)]  # Counter for honest observations in leaf i
        tmp_y_t = [0 for i in range(self.tree.tree_.node_count)]  # Counter for positive honest observations in leaf i
        tmp_n_c = [0 for i in range(self.tree.tree_.node_count)]  # Counter for honest observations in leaf i
        tmp_y_c = [0 for i in range(self.tree.tree_.node_count)]  # Counter for positive honest observations in leaf i
        for i, item in enumerate(leaf_idx):
            if honest_y[i]:
                if honest_t[i]:
                    tmp_y_t[item] += 1
                else:
                    tmp_y_c[item] += 1
            if honest_t[i]:
                tmp_n_t[item] += 1
            else:
                tmp_n_c[item] += 1
        # Loop through tmp to populate nodes with new estimates.
        # Hmm, leaf_idx is the idx of the nodes. The numbering is
        # hence not 0:n_leaves, but has gaps for the internal nodes
        # in the tree.
        # Make a dictionary and use idx as key? That would at least
        # be fast as dicts internally use a hash function.
        for i in range(self.tree.tree_.node_count):
            # -Calculate n_t, y_t, n_c, and y_c for all nodes.
            n = tmp_n_c[i] + tmp_n_t[i]
            if n == 0:
                # No observations (probably not leaf)
                self.node_dict[str(i)] = None
            else:
                alpha_1 = tmp_y_t[i] + 1
                beta_1 = tmp_n_t[i] - tmp_y_t[i] + 1
                alpha_0 = tmp_y_c[i] + 1
                beta_0 = tmp_n_c[i] - tmp_y_c[i] + 1
                tmp_hpd = beta_difference.uncertainty(alpha_1, beta_1, alpha_0, beta_0)
                try:
                    tau = (tmp_y_t[i] / tmp_n_t[i]) -\
                        (tmp_y_c[i] / tmp_n_c[i])
                except:
                    print("Problem in honest estimation")
                    print(tmp_y_t[i])
                    print(tmp_n_t[i])
                    print(tmp_y_c[i])
                    print(tmp_n_c[i])
                # It makes a lot of sense to estimate hpd things at this point.
                self.node_dict[str(i)] = {'tau': tau, 'hpd': tmp_hpd, 'alpha_t': alpha_1,
                                          'beta_t': beta_1, 'alpha_c': alpha_0,
                                          'beta_c': beta_0}
                # -Estimate \hat{p} and HPD-intervals for all leafs

    def predict_uplift(self, X):
        """
        Args:
        X (np.array): Features of observation(s)
        """
        nodes = self.tree.apply(X)
        # Fetch parameters for corresponding leafs
        tmp = [self.node_dict[str(item)] for item in nodes]
        return tmp
        # Return as list of dicts?

    def undersampling_fit(self, data, k_values=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
        """
        Fit-method that handles undersampling and searches for optimal undersampling
        parameter k using best AUUC on the training set. This will cause some 
        overfitting, but the leaf parameters are finally evaluated using the
        validation set that had been untouched until that point ensuring honest
        estimation of the leaf parameters.

        Args:
        data (load_data.DatasetCollection): Standard format to make things
         easier...
        k_values (list(float or int)): k-parameters to search over
        """
        best_auuc = -1.0
        # 1. Loop over k's
        for k in k_values:
            # 2. Undersample training set
            # Go with stratified undersampling
            try:
                tmp_data = data.undersampling(k)
            except AssertionError:
                print("Not enough observations for k {}. Moving on...".format(k))
                continue
            tmp_model = HonestUpliftTree(1000)
            tmp_model.fit(tmp_data['X'], tmp_data['r'])
            tmp_model.honest_fit(data['training_set']['X'], data['training_set']['y'],
                                 data['training_set']['t'])
            # 3. Estimate AUUC on training set
            try:
                tmp_pred = tmp_model.predict_uplift(data['training_set']['X'])
            except AssertionError:
                print("Could not do honest fit for k {}".format(k))
                continue
            tmp_pred = np.array([item['tau'] for item in tmp_pred])
            # The metrics package ends up throwing an error when estimating
            # EUCE and MUCE as the selected k is too large... Set to 2?
            tmp_metrics = uplift_metrics.UpliftMetrics(data['training_set']['y'],
                                                       tmp_pred,
                                                       data['training_set']['t'],
                                                       k=2)
            # 4. See if model is best so far
            if tmp_metrics.auuc > best_auuc:
                best_model = deepcopy(tmp_model)
                best_auuc = tmp_metrics.auuc
                best_k = k
        # 5. Store or return model? Or could all of this just be a method of
        # the previous class?
        # 5.1 Set the tree to the best learned structure
        self.tree = best_model.tree
        # 5.2 Estimate new honest parameters from the validation set
        self.honest_fit(data['validation_set']['X'], data['validation_set']['y'],
                        data['validation_set']['t'])
        # 5.3 Store the best k
        self.k = best_k

    def honest_undersampling_fit(self, data, k_values=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                                 cv_folds=5):
        """
        Same as above, except that parameter estimation is done in an honest fashion,
        and this method splits the training set into training and calibration sets.
        (Some handling of the data-object. Maybe pass 'size' to deal with it?)
        Requires that subsets 'tree_train' and 'tree_val' have been added to the
        DatasetCollection.

        Fit-method that handles undersampling and searches for optimal undersampling
        parameter k using best AUUC on the training set. This will cause some 
        overfitting, but the leaf parameters are finally evaluated using the
        validation set that had been untouched until that point ensuring honest
        estimation of the leaf parameters.

        Args:
        data (load_data.DatasetCollection): Standard format to make things
         easier...
        k_values (list(float or int)): k-parameters to search over
        cv_folds (int): Number of folds to use in cross-validation for selection of
         best k.
        """
        best_auuc = -1.0
        # 1. Loop over k's
        for k in k_values:
            # 2. Undersample training set
            # Go with stratified undersampling
            try:
                tmp_data = data.undersampling(k, target_set='tree_train')
            except AssertionError:
                print("Not enough observations for k {}. Moving on...".format(k))
                continue
            n_samples = tmp_data['X'].shape[0]
            fold_idx = [[i + j * n_samples // cv_folds for i in range(n_samples // cv_folds)] for j in range(cv_folds)]
            tmp_metrics = []  # Reset metrics for new k
            for i in range(cv_folds):
                # N-fold cross-validation
                tmp_fold_idx = [item for j, item in enumerate(fold_idx) if j != i]
                # Unpack:
                train_idx = [item for item in itertools.chain.from_iterable(tmp_fold_idx)]
                cross_val_idx = fold_idx[i]
                X = tmp_data['X'][train_idx, :]
                r = tmp_data['r'][train_idx]
                tmp_model = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                                                  min_samples_leaf=self.min_samples_leaf)
                tmp_model.fit(X, r)
                # Evaluate model on hold-out set:
                X_val = tmp_data['X'][cross_val_idx, :]
                y_val = tmp_data['y'][cross_val_idx]
                t_val = tmp_data['t'][cross_val_idx]
                tmp_pred = tmp_model.predict(X_val)
                tmp_metric = uplift_metrics.UpliftMetrics(y_val, tmp_pred, t_val, k=2)
                tmp_metrics.append(tmp_metric)
            # Average metrics over folds
            tmp_auuc = np.average([item.auuc for item in tmp_metrics])
            # 4. See if model is best so far
            if tmp_auuc > best_auuc:
                best_auuc = tmp_auuc
                best_k = k
        # At this point we have found the best k to use for model training.
        # Next training with full training data:
        tmp_data = data.undersampling(best_k, target_set='tree_train')
        self.tree.fit(tmp_data['X'], tmp_data['r'])
        # 5.2 Estimate new honest parameters from the validation set
        self.honest_fit(data['tree_val']['X'], data['tree_val']['y'],
                        data['tree_val']['t'])
        # 5.3 Store the best k
        self.k = best_k

    def generate_sample(self, X_item, N_observations=1000):
        """
        Methods for generating observations sampled from the assumed beta-difference
        distribution with the learned parameters.

        Args:
        X_item (np.array): A 2-d array of features for _one_ observations.
        N_observations (int): Number of observations to draw in Monte Carlo
         simulation.
        """
        tmp = self.predict_uplift(X_item)
        # Maybe change format for how predict returns items? Really strange to pick one observation
        # like this:
        try:
            t_observations = beta.rvs(a=tmp[0]['alpha_t'], b=tmp[0]['beta_t'], size=N_observations)
            c_observations = beta.rvs(a=tmp[0]['alpha_c'], b=tmp[0]['beta_c'], size=N_observations)
            tau_observations = t_observations - c_observations
        except:
            print(tmp[0]['alpha_t'])
            print(tmp[0]['beta_t'])
            print(tmp[0]['alpha_c'])
            print(tmp[0]['beta_c'])
        return tau_observations

    def estimate_mean_credible_interval_width(self, X, p_mass=95):
        """
        Method for estimating average credible interval width. While this is
        a metric, it makes more sense to keep it here than in the metrics
        package, as calculation of this metric is dependent on the tree.

        Args:
        X (np.array): Features of the observations
        """
        if p_mass == 95:
            # Use existing predictions if
            tmp = self.predict_uplift(X)
            return np.mean([item['hpd']['width'] for item in tmp])
        else:
            predictions = self.predict_uplift(X)
            hpd_list = []
            for row in predictions:
                    tmp_hpd = beta_difference.uncertainty(
                        predictions['alpha_t'], predictions['beta_t'],
                        row['alpha_c'], row['beta_c'], p=p_mass)
                    hpd_list.append(tmp_hpd['width'])
            return np.mean(hpd_list)
