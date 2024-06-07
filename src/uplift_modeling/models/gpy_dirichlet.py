"""
Code to implement the exact Dirichlet-based Gaussian Process (Milios & al)
using GPytorch following the example on 
https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html
"""


from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
import torch
import gpytorch
import numpy as np
import copy
import gc

if False:  # Code does not support any GPU acceleration yet.
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DirichletGP():
    """This wraps class above neatly.

    """

    def __init__(self):
        self.likelihood = None  # Likelihood of _transformed_ _parameter_!!
        self.model = None
        self.mean_negative_log_likelihood = None  # Negative log-likelihood of training data
        self.negative_log_likelihood = None

    def fit(self, X, y, alpha_eps=0.1, training_iter=10):
        """First, do necessary variable transformation of y

        Parameters
        ----------
        X : np.array
            Features
        y : np.array
            Labels
        """

        train_y = np.array([item * 1 for item in y])
        train_y = torch.from_numpy(train_y)
        # train_y.to(DEVICE)
        train_x = torch.from_numpy(X)
        # train_x.to(DEVICE)
        # initialize likelihood and model
        # we let the DirichletClassificationLikelihood compute the targets for us
        self.likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True,
                                                            alpha_epsilon=alpha_eps)
        # self.likelihood.transformed_targets.to(DEVICE)
        # self.likelihood.to(DEVICE)
        self.model = DirichletGPModel(train_x, self.likelihood.transformed_targets, 
                                      self.likelihood, num_classes=self.likelihood.num_classes)
        # self.likelihood.to(DEVICE)
        # self.model.to(DEVICE)
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        last_loss = 1e6  # "Big number"
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x) # self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.likelihood.transformed_targets).sum()
            loss.backward()
            if i % 5 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.mean().item(), 
                    self.model.likelihood.second_noise_covar.noise.mean().item()))
            optimizer.step()
            if last_loss <= loss:
                # Loss has not improved over five iterations.
                break
            else:
                last_loss = loss
            
        # Estimate negative log-likelihood for training data
        # (not transformed labels, but actual labels).
        # This is following the advice of Milios & al.
        predictions = self.predict(X)  # Array with two columns, one for negative class, one for positive
        predictions = np.log(predictions)
        negative_log_likelihood = 0
        for i, item in enumerate(y):
            if item:
                negative_log_likelihood -= predictions[1][i]
            else:
                negative_log_likelihood -= predictions[0][i]
        self.negative_log_likelihood = negative_log_likelihood
        mean_negative_log_likelihood = negative_log_likelihood / y.shape[0]
        self.mean_negative_log_likelihood = mean_negative_log_likelihood

    def predict(self, X):
        """Method for predicting probabilities.

        """

        # Predictions
        self.model.eval()
        self.likelihood.eval()
        test_x = torch.from_numpy(X)
        #test_x.to(DEVICE)

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.model(test_x)
            pred_means = test_dist.loc

        # Sampling
        pred_samples = test_dist.sample(torch.Size((256,))).exp()
        probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        #probabilities.to('cpu')
        probabilities = probabilities.numpy()
        return probabilities

    def generate_sample(self, X, sample_size=1000):
        """
        Generate observations from distribution of uncertainty for _one_
        observation at a time.
        Returns 2-d array with probability of negative class in first column
        and probability of positive class in second column.

        Parameters
        ----------
        X : np.array
            Features of _one_ observation.
        sample_size : int
            Number of observations to generate. Milios used 1000, gpytorch example 256.
        """

        self.model.eval()
        self.likelihood.eval()

        test_x = torch.from_numpy(X.reshape(1, -1))
        #test_x.to(DEVICE)

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.model(test_x)
            pred_means = test_dist.loc

        # Sampling
        pred_samples = test_dist.sample(torch.Size((sample_size,))).exp()
        # Normalizaion:
        samples = pred_samples.numpy()
        samples = samples.squeeze()  # Remove third dimension
        samples = samples / samples.sum(1).reshape(-1, 1)  # Samples already exponentiated
        return samples


class DirichletGPUplift():
    """Uplift model based on the gpytorch-implementation.
    """

    def __init__(self):
        self.model_t = DirichletGP()
        self.model_c = DirichletGP()
        self.mean_negative_log_likelihood = None
        self.alpha_epsilon = None

    def fit(self, X_t, y_t, X_c, y_c, alpha_eps=0.1, max_iterations=1000, auto_alpha_eps=True):
        """
        Parameters
        ----------
        X_t : np.array
            Features
        y_t : np.array
            Features
        alpha_eps : float
            Parameter for variable transformation
        max_iterations : int
            Maximum number of training iterations
        auto_alpha_eps : bool
            Automatically search for good alpha_eps. Setting this to 
            True will ignore alpha_eps parameter.
        """

        t_observations = y_t.shape[0]
        c_observations = y_c.shape[0]
        tot_observations = t_observations + c_observations
        best_mean_neg_log_likelihood = np.inf
        if auto_alpha_eps:
            alpha_eps_list = [2**(-i) for i in range(8)]
        else:
            alpha_eps_list = [alpha_eps]
        for alpha_eps in alpha_eps_list:
            # Train model, store log-likelihood.
            # Keep model with best log-likelihood
            # Perhaps check whether that log-likelihood is at
            # either end of tried parameter list suggesting that
            # optimum might be further away.
            tmp_model_t = DirichletGP()
            tmp_model_c = DirichletGP()
            tmp_model_t.fit(X_t, y_t, alpha_eps=alpha_eps, training_iter=max_iterations)
            tmp_model_c.fit(X_c, y_c, alpha_eps=alpha_eps, training_iter=max_iterations)
            tmp_mean_neg_log_likelihood = (tmp_model_t.negative_log_likelihood +\
                tmp_model_c.negative_log_likelihood) / tot_observations
            print("Alpha_epsilon: {}, MNLL: {}".format(alpha_eps, tmp_mean_neg_log_likelihood))
            if tmp_mean_neg_log_likelihood < best_mean_neg_log_likelihood:
                # Store model in self
                best_mean_neg_log_likelihood = copy.deepcopy(tmp_mean_neg_log_likelihood)
                best_model_t = copy.deepcopy(tmp_model_t)
                best_model_c = copy.deepcopy(tmp_model_c)
                best_alpha_epsilon = alpha_eps
            del tmp_model_t
            del tmp_model_c
            del tmp_mean_neg_log_likelihood
            gc.collect()
        self.model_t = best_model_t
        self.model_c = best_model_c
        self.alpha_epsilon = best_alpha_epsilon
        self.mean_negative_log_likelihood = best_mean_neg_log_likelihood
        # else:
        #     # Regular fit with just one value in alpha_eps
        #     self.model_t.fit(X_t, y_t, alpha_eps=alpha_eps, training_iter=max_iterations)
        #     self.model_c.fit(X_c, y_c, alpha_eps=alpha_eps, training_iter=max_iterations)
        #     self.mean_negative_log_likelihood = (self.model_t.negative_log_likelihood +\
        #         self.model_c.negative_log_likelihood) / tot_observations
        print("Optimal mean negative log-likelihood {} with alpha_epsilon {}".format(
            self.mean_negative_log_likelihood, self.alpha_epsilon))

    def predict_uplift(self, X):
        """Predict method.
        """

        tmp_t = self.model_t.predict(X)
        tmp_c = self.model_c.predict(X)
        # tmp_t[1] contains probabilities for positive class
        uplift = tmp_t[1] - tmp_c[1]
        return uplift

    def generate_sample(self, X, sample_size=256):
        """
        Method for generating observations from uncertainty distribution
        conditional on features X.

        Args:
        X (np.array): Features of _ONE_ observation.
        sample_size (int): Number of observations to generate.
        """

        tmp_t = self.model_t.generate_sample(X, sample_size=sample_size)
        tmp_c = self.model_c.generate_sample(X, sample_size=sample_size)
        # Defining uplift as the difference between probabilities for positive outcome:
        tau = tmp_t[:, 1] - tmp_c[:, 1]
        return tau

    def get_credible_intervals(self, X, p_mass=0.95, sample_size=1000):
        """
        Method for getting HPD credible intervals for all of X.

        Args:
        X (np.array): Features of observations
        p_mass (float): In ]0, 1]. Probability mass that needs to
        fall within credible interval.
        sample_size (int): Size of sample to generate in MC.
        """

        # Auxiliary function
        def find_smallest_window(tau, p_mass=0.95):
            """
            Args:
            tau (np.array): Array of ... WHAT? Samples from distribution.
            Do they have to be in [0, 1]? E.g. a GP could potentially
            produce something else.
            """

            # 1. Sort tau in increasing order
            tau = np.sort(tau)
            # 2. Calculate window size N_{1-alpha}
            sample_size = len(tau)
            n_interval = int(sample_size * p_mass)
            lower_idx = 0
            upper_idx = n_interval
            # In some skewed cases, the HPD interval starts at the smallest observation:
            smallest_low_idx = 0  # Initial values
            smallest_up_idx = upper_idx  # Initial values
            # 3. Estimate width of sliding window
            smallest_width = tau[smallest_up_idx] - tau[smallest_low_idx]
            for lower_idx in range(sample_size - n_interval):
                upper_idx = lower_idx + n_interval
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

        # 1. Find unique X (is this fast? At least for 10k observations.)
        # At least in Hillstrom, like 90% of all observations are unique.
        # Skip this step.
        # 2. Estimate width for every unique X
        intervals = []
        for i, _ in enumerate(X):
            samples = self.generate_sample(X[i, :], sample_size=sample_size)
            interval = find_smallest_window(samples, p_mass)
            intervals.append(interval)
        return intervals

    def mean_credible_interval_width(self, X, p_mass=0.95, mc_samples=1000):
        """
        Method for estimating the average width of the credible intervals.

        Args:
        X (np.array): Features
        p_mass (float): In ]0, 1]. Probability mass in HPD-interval.
        """

        credible_intervals = self.get_credible_intervals(X, p_mass, mc_samples)
        return np.mean([item['width'] for item in credible_intervals])
