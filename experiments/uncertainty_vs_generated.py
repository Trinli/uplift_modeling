"""
This file will contain code for verifying that the uncertainty models developed
for uplift actually produce valid uncertainty estimates. We are specifically
dealing with epistemic uncertainty, i.e. the uncertainty that is a result of
having only limited data to train on.

One way to test this is by first defining a generating model with ground truth,
then generate data using it, estimating how well the uncertainty match the
generating parameters (i.e. not the actual label, but the distribution parameters
from which the observation was drawn).
Do this N times to get some sort of distribution. 19/20 should fit into the
95% credible intervals.
"""
import random
import numpy as np
import data.load_data as ld
import metrics.uplift_metrics as um
import models.honest_uplift_tree as honest_tree
import models.gpy_dirichlet as GPY


data_format = ld.STARBUCKS_FORMAT  # Change SEED in data format to re-randomize.
data_format['file_name'] = './datasets/' + data_format['file_name']

def main(model='tree'):
    honest = True
    seed = random.randint(0, 2**16)
    print("Using seed: {}".format(seed))
    data_format['random_seed'] = seed
    data = ld.DatasetCollection(data_format['file_name'], data_format=data_format)
    size = 32000
    # Add suitable sets
    if model == 'tree':
        if honest:
            data.add_set('tree_train', 0, int(size/2))
            data.add_set('tree_val', int(size/2), size)
        else:
            data.add_set('tree_train', 0, size)
    elif model == 'gp' or model == 'dgp':
        data.add_set('gp_train', 0, size)


    # Actually, we want to generate data. 
    # How?
    # -Semi-synthetic? Features from e.g. Criteo, create a model to generate
    # class-labels from features and treatment label. Model has to be such
    # that it produces the parameters for some probability distribution (Bernoulli?)
    # and then the labels are drawn from that.
    # Starbucks has 7 features. Model can be something like:
    #  theta_i = sigmoid( (x_1 * a * x_2 * b) + x_3 * c + x_4 * d + x_5 * e + (x_6 * f * x_7 * g) * t )
    #  y_i ~ Bernoulli(theta_i)
    # Here the last term is a product with t making the uplift something else than an additive term.
    def theta(x, t):
        """
        Function estimates the bernoulli parameter given the features and
        draws a random sample from that distribution as label.
        It returns both the theta and the y.

        x (np.array): A 7-dimensional vector for Starbucks data.
        """
        def sigmoid(a):
            return 1 / (1 + np.exp(a))
        
        # This is the weight vector. Just picked something. 
        # Maybe draw from a random normal? Draw should be performed only once.
        w = [2, .3, .7, -1.5, -.2, 11.2, 5.3]  # Last two are 'big' to result in some sort of uplift...
        tmp = (w[0] * x[0] * w[1] * x[1]) + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + (w[5] * x[5] * w[6] * x[6]) * t
        theta_i = sigmoid(tmp)
        tmp_t = (w[0] * x[0] * w[1] * x[1]) + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + (w[5] * x[5] * w[6] * x[6])
        theta_t = sigmoid(tmp_t)
        tmp_c = (w[0] * x[0] * w[1] * x[1]) + w[2] * x[2] + w[3] * x[3] + w[4] * x[4]
        theta_c = sigmoid(tmp_c)
        y_i = np.random.binomial(1, theta_i, 1)[0].astype(bool)  # Extract value from array.
        return {'theta': theta_i, 'y': y_i, 'theta_t': theta_t, 'theta_c': theta_c}

    def theta_vec(X, t):
        """
        X (np.array): Matrix of features
        t (np.array): Vector of treatment labels
        """
        tmp = np.array([theta(item, t_item) for item, t_item in zip(X, t)])
        y = np.array([item['y'] for item in tmp])
        theta_i = np.array([item['theta'] for item in tmp])
        theta_t = np.array([item['theta_t'] for item in tmp])
        theta_c = np.array([item['theta_c'] for item in tmp])
        theta_tau = theta_t - theta_c
        return {'y': y, 'theta': theta_i, 'theta_t': theta_t, 'theta_c': theta_c, 'theta_tau': theta_tau}

    def revert_label(y_vec, t_vec):
        """
        Class-variable transformation ("revert-label") approach
        following Athey & Imbens (2015).

        Args:
        y_vec (np.array([float])): Array of conversion values in samples.
        t_vec (np.array([bool])): Array of treatment labels for same samples.
        """
        N_t = sum(t_vec == True)
        N_c = sum(t_vec == False)
        # N_t = sum(t_vec)
        # N_c = t_vec.shape[0] - N_t
        # This needs to be sorted out.
        p_t = N_t / (N_t + N_c)
        assert 0.0 < p_t < 1.0, "Revert-label cannot be estimated from only t or c observations."
        def revert(y_i, t_i, p_t):
            return (y_i * (int(t_i) - p_t) / (p_t * (1 - p_t)))
        r_vec = np.array([revert(y_i, t_i, p_t) for y_i, t_i in zip(y_vec, t_vec)])
        #r_vec = r_vec.astype(self.data_format['data_type'])
        return r_vec


    # New_labels will contain true label as 'y' and model parameter as 'theta'
    if model == 'tree':
        X = data['tree_train']['X']
        t_ = data['tree_train']['t']
        t = np.random.binomial(1, p=.5, size=t_.shape[0]).astype(bool)
        new_labels = theta_vec(X, t)
        r = revert_label(new_labels['y'], t)

        X_honest = data['tree_val']['X']
        t_honest_ = data['tree_val']['t']
        t_honest = np.random.binomial(1, p=.5, size=t_honest_.shape[0]).astype(bool)
        new_honest_labels = theta_vec(X_honest, t_honest)

        tree_model = honest_tree.HonestUpliftTree(max_leaf_nodes=12, min_samples_leaf=100)  # Previous experiments used max_leaf_nodes=12 for Starbucks
        tree_model.fit(X, r, honest_X=X_honest, honest_y=new_honest_labels['y'], honest_t=t_honest,
                    y=new_labels['y'], t=t)

        X_test = data['testing_set']['X']
        t_test_ = data['testing_set']['t']
        t_test = np.random.binomial(1, p=.5, size=t_test_.shape[0]).astype(bool)
        y_test = theta_vec(X_test, t_test)  # This now contains the parameters generating the y values.
        # We should actually generate both y_{i, t=0} and y_{i, t=1} to know the true uplift.
        # "Generate" t. 0 and 1 for every observation. Maybe randomize the t for training.
        predictions = tree_model.predict_uplift(X_test)
        tau_pred = np.array([item['tau'] for item in predictions])
        # Sanity check
        metrics = um.UpliftMetrics(y_test['y'], tau_pred, t_test)
        print(metrics)

        # for i in range(10):
        #     print('=' * 40)
        #     print("Generating value (theta_tau): {}".format(y_test['theta_tau'][i]))
        #     print("Predicted value: {}".format(tau_pred[i]))
        #     print("Estimated CI lower: {} \t upper: {}".format(predictions[i]['hpd']['lower_bound'], predictions[i]['hpd']['upper_bound']))

        counter = 0
        for i, item in enumerate(predictions):
            if item['hpd']['lower_bound'] <= y_test['theta_tau'][i]:
                if item['hpd']['upper_bound'] >= y_test['theta_tau'][i]:
                    counter += 1
        generating_ite_within_ci = counter / len(predictions)
        print("{}% of the predictions are within 95% HPD-interval".format(100 * generating_ite_within_ci))

        tree_average_width = tree_model.estimate_mean_credible_interval_width(X_test)
        print("Average CI: {}".format(tree_average_width))
        # >>> print("Average CI: {}".format(tree_average_width))  # This with 80 leaves.
        np.mean(y_test['theta_tau'])
        # Notes
        # Average generating theta_tau within a leaf should fall within the 95% CI 19 times out of 20.
        # This may be reasonable in a tree where every observation gets the same uncertainty estimates
        # and the "local neighborhood" of x is well-defined, constant.
        # I don't see how this could be generalized to DGP.
        # theta_tau_i is the INDEPENDENT TREATMENT EFFECT (ITE), not CATE!
        # For DGP, maybe put observations into bins (how do we get narrow enough bins?) and compare the
        # bin average theta_tau_i to the predicted CI:s observation by observation?

        # Start by arranging observations by leafs. Can there be two leaves with identical output? 
        unique_predictions = np.unique([item['tau'] for item in predictions])
        within_ci_by_leaf = []
        for pred in unique_predictions:
            items_in_leaf = []
            generating_tau_theta = []
            for i, item in enumerate(predictions):
                if pred == item['tau']:
                    # Append boundaries for item
                    # Append generating parameter
                    # Check if average of generating parameters are within boundaries prediction by prediction
                    generating_tau_theta.append(y_test['theta_tau'][i])
                    items_in_leaf.append(item)
            avg_tau_theta = np.mean(generating_tau_theta)
            print("Leaf with prediction {}".format(pred))
            print("Average generating parameter: {}".format(avg_tau_theta))
            count = 0
            for item in items_in_leaf:
                if item['hpd']['lower_bound'] <= avg_tau_theta:
                    if item['hpd']['upper_bound'] >= avg_tau_theta:
                        count += 1
            print("Leaf contains {} observations".format(len(items_in_leaf)))
            fraction_within_ci = count / len(items_in_leaf)
            print("{} of observations in leaf were within predicted CI's".format(fraction_within_ci))
            within_ci_by_leaf.append([fraction_within_ci, len(items_in_leaf)])
        within_ci_overall = np.average([item[0] for item in within_ci_by_leaf], weights=[item[1] for item in within_ci_by_leaf])
        print("{}% of generating mean(theta_tau_i) per leaf is within the predicted credible intervals".format(within_ci_overall))
        return within_ci_overall, generating_ite_within_ci

    elif model == 'dgp':
        X = data['gp_train']['X']
        t_ = data['gp_train']['t']
        t = np.random.binomial(1, p=.5, size=t_.shape[0]).astype(bool)
        new_labels = theta_vec(X, t)
        X_t = X[t, :]
        y_t = new_labels['y'][t]
        X_c = X[~t, :]
        y_c = new_labels['y'][~t]

        dgp_model = GPY.DirichletGPUplift()
        dgp_model.fit(X_t, y_t, X_c, y_c, alpha_eps=.1, auto_alpha_eps=True)
        X_test = data['testing_set']['X']
        t_test_ = data['testing_set']['t']
        t_test = np.random.binomial(1, p=.5, size=t_test_.shape[0]).astype(bool)
        y_test = theta_vec(X_test, t_test)  # This now contains the parameters generating the y values.
        predictions = dgp_model.predict_uplift(X_test)
        uncertainty = dgp_model.get_credible_intervals(X_test)
        tau_pred = predictions
        # Sanity check
        counter = 0
        for i, item in enumerate(uncertainty):
            if item['lower_bound'] <= y_test['theta_tau'][i]:
                if item['upper_bound'] >= y_test['theta_tau'][i]:
                    counter += 1
        theta_gen_ite_in_ci = counter / len(predictions)
        print("{}% of the predictions are within 95% HPD-interval".format(100 * theta_gen_ite_in_ci))
        # >>> print("{}% of the predictions are within 95% HPD-interval".format(counter/len(predictions)))
        # 0.8111646484498827% of the predictions are within 95% HPD-interval  # 81% (not 0.81%...)
        # This indicates that the CATE is a good proxy for the ITE in this case, although the model assumed
        # is not particularly complex. Hard to say how well this generalizes to "real data" (which cannot
        # be tested as we don't know the ground truth).
        dgp_average_width = dgp_model.mean_credible_interval_width(X_test, 0.95)
        test_description = 'theta gen ite in 95% CI: {}, dgp_average_width: {}'.format(theta_gen_ite_in_ci, dgp_average_width)
        metrics = um.UpliftMetrics(y_test['y'], tau_pred, t_test, test_description=test_description)
        metrics.write_to_csv('./results/generated_theta_ITE_vs_CI_dgp.csv')
        print(metrics)
        return theta_gen_ite_in_ci


res = []
for i in range(10):  # 400 was a bit of overkill...
    res.append(main('tree'))  # Checking how well the generating parameters of the leafs fall within estimated 95% leaf-CI's
    # Should we also check how often the generating parameters of individual observations fall within the 95% CI's?

res_dgp = []
for i in range(10):
    res_dgp.append(main('dgp'))

