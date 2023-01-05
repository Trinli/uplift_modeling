"""
Dirichlet-based Gaussian Process for classification following
Milos & al. 2018 
**** AND ****
An uplift model using said GPD as base-learner. This uplift
model is our contribution. It contains methods to evaluate
the uncertainty of the estiamtes.

Code based on https://github.com/dmilios/dirichletGPC

Misc:
Maybe test whether newer versions of gpflow would work that are
also compatible with newer versions of tensorflow.

Big downside is that this code requires old version of tensorflow
because it also uses an old version of gpflow. Maybe the used
gpflow can be upgraded and tensorflow as well?
"""
import numpy as np
import gpflow
import sys
import time
sys.path.insert(0,'../dirichletGPC/src')
# sys.path.append("/Users/notto/Desktop/dirichletGPC/src")
import heteroskedastic


class DirichletGP():
    """
    Dirichlet-based Gaussian Process following Milos & al. 2018.
    Only implements binary label.
    Original code at https://github.com/dmilios/dirichletGPC
    """

    def __init__(self):
        """
        Args:
        a_eps (float): "Bias" for dirichlet priors. Must be larger than 0.
        """
        self.model = None  # Needed outside of fit
        self.y_mean = None  # Needed outside of fit
        self.report = {}

    def fit(self, X, y, Z=None,  a_eps=0.1,
            ARD=False, ampl=None, leng=None):
        """
        Perhaps split into "get_parameters()" and "estimate_density()"
        or something. Would be useful to extract the parameters for
        further post-processing.

        Args
        X (np.array): Features
        y (np.array): Binary labels
        """
        dim = X.shape[1]
        if ARD:
            len0 = np.repeat(np.mean(np.std(X, 0)) * np.sqrt(dim), dim)
        else:
            len0 = np.mean(np.std(X, 0)) * np.sqrt(dim)

        X = X.astype(np.float64)
        # Reshape y to a 2-dim vector
        y = np.array([[int(item == False), int(item == True)] for item in y]).astype(np.float64)
        # Variance:
        s2_tilde = np.log(1.0 / (y + a_eps) + 1)
        s2_tilde = s2_tilde.astype(np.float64)
        y_tilde = np.log(y + a_eps) - 0.5 * s2_tilde
        self.y_mean = np.log(y.mean(0)) + np.mean(y_tilde - np.log(y.mean(0)))
        y_tilde = y_tilde - self.y_mean
        y_tilde = y_tilde.astype(np.float64)

        # Define model
        var0 = np.var(y_tilde)
        kernel = gpflow.kernels.RBF(dim, ARD=ARD, lengthscales=len0,
                                    variance=var0)

        if Z is None:
            Z = X

        self.model = heteroskedastic.SGPRh(X, y_tilde, kern=kernel,
                                           sn2=s2_tilde, Z=Z)
        # Initialize optimizer
        opt = gpflow.train.ScipyOptimizer()
        if ampl is not None:
            kernel.variance.trainable = False
            kernel.variance = ampl * ampl
        if leng is not None:
            kernel.lengthscales.trainable = False
            if ARD:
                kernel.lengthscales = np.ones(dim) * leng
            else:
                kernel.lengthscales = leng

        # Optimize model:
        gpd_elapsed_optim = None
        if ampl is None or leng is None or a_eps is None:
            print('gpd optim... ', end='', flush=True)
            start_time = time.time()
            opt.minimize(self.model)
            gpd_elapsed_optim = time.time() - start_time
            print('done!')
            self.report['gpd_elapsed_optim'] = gpd_elapsed_optim

        gpd_amp = np.sqrt(self.model.kern.variance.read_value())
        self.report['gpd_amp'] = gpd_amp
        gpd_len = self.model.kern.lengthscales.read_value()
        self.report['gpd_len'] = gpd_len
        self.report['gpd_a_eps'] = a_eps

    def predict_params(self, X):
        """
        A predict function that returns the predicted parameters
        rather than e.g. the mean and upper and lower 95% quantiles.
        This makes post-processing possible, e.g. the distribution
        of the uncertainty of a prediction can be plotted in full.

        Args:
        X (np.array): Features.
        """
        fmu_original, fs2 = self.model.predict_f(X.astype(np.float64))
        fmu = fmu_original + self.y_mean
        return {'fmu': fmu, 'fs2': fs2}

    def uncertainty_of_prediction(self, fmu_item, fs2_item, N_samples=1000):
        """
        Method to create histogram of distribution of uncertainty
        for _one_ prediction.
        This should not actually be part of the class in this format.
        Does not need model or any other parameters in any way.
        -Auxiliary function?
        Perhaps take X as argument (one observation) and estimate parameters
        from that etc. 
        The outputs are samples from the distribution of negative and positive
        predictions. Generally we are interested in the distribution of the
        positive class.
        Outputs samples from ... what? Why is it 2-d? They are samples for class
        0 and class 1. Class 1 is what we are interested in. As in negative and
        positive class.

        Args:
        fmu (np.array): Mean values for class 0 and class 1.
        fs2 (np.array): Variances for class 0 and class 1
        mc_samples (int): Number of observations to draw in Monte
         Carlo simulation.
        """
        source = np.random.randn(N_samples, 2)  # Sampling random normal. Binary case.
        # Generating observations of new distribution (approximated gamma)
        # samples = source * np.sqrt(fs2[i,:]) + fmu[i,:]
        samples = source * np.sqrt(fs2_item) + fmu_item
        # Using the gamma observations to create the Dirichlet distribution
        samples = np.exp(samples) / np.exp(samples).sum(1).reshape((-1, 1))
        return samples

    def predict(self, X, q=95):
        """
        Method now returns mean prediction and upper and lower q-percentile
        bounds. Note that this is not an HPD-interval!!!

        Args:
        X (np.array): Features for observations to predict from.
        q (float): Percentile (i.e. in 0-100) for desired upper
         and lower bounds.
        """
        # Prediction
        tmp = self.predict_params(X)
        fmu = tmp['fmu']
        fs2 = tmp['fs2']

        # Estimate mean and quantiles of the Dirichlet distribution through sampling
        # q=95
        mu_dir = np.zeros([fmu.shape[0], 2])
        lb_dir = np.zeros([fmu.shape[0], 2])
        ub_dir = np.zeros([fmu.shape[0], 2])
        
        source = np.random.randn(1000, 2)  # Sampling random normal
        random_sample = []
        for i in range(fmu.shape[0]):
            # Generating observations of new distribution (approximated gamma)
            samples = source * np.sqrt(fs2[i,:]) + fmu[i,:]
            # Using the gamma observations to create the Dirichlet distribution
            samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
            # Estimating distribution
            Q = np.percentile(samples, [100-q, q], axis=0)
            mu_dir[i,:] = samples.mean(0)
            lb_dir[i,:] = Q[0,:]
            ub_dir[i,:] = Q[1,:]
            random_sample.append(samples[0])  # Add one randomly drawn sample from the distribution
        return {'mu': mu_dir, 'lb': lb_dir, 'ub': ub_dir, 'samples': random_sample}

    def generate_sample(self, X):
        """
        Method that generates a random sample for every observation in X from
        the conditional distribution on x.

        Args:
        X (np.array): Features for observations to predict from.
        """
        # Prediction
        tmp = self.predict_params(X)
        fmu = tmp['fmu']
        fs2 = tmp['fs2']

        source = np.random.randn(1, 2)  # Sampling random normal
        generated_samples = []
        for i in range(fmu.shape[0]):
            # Generating observations of new distribution (approximated gamma)
            samples = source * np.sqrt(fs2[i,:]) + fmu[i,:]
            # Using the gamma observations to create the Dirichlet distribution
            samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
            # Add to list:
            generated_samples.append(samples[0][1])  # Add one sample from the distribution
        return generated_samples



class DirichletUplift():
    """
    An uplift model using the Dirichlet-based Gaussian Process
    as base learner.

    Maybe set up class so that the uncertainties can be predicted
    or plotted for one sample?
    We might need to implement search over a_eps.
    """
    def __init__(self):
        self.model_t = DirichletGP()
        self.model_c = DirichletGP()

    def fit(self, X_t, y_t, X_c, y_c, a_eps=0.1):
        """
        Args:
        X_t (np.array): Features of _treated_ observations
        y_t (np.array): Labels of treated observations
        X_c (np.array): Features of untreated (control) observations
        y_c (np.array): Labels of untreated observations
        """
        self.model_t.fit(X_t, y_t, a_eps=a_eps)
        self.model_c.fit(X_c, y_c, a_eps=a_eps)

    def predict_uplift(self, X):
        """
        Bla?
        Focus on positive label?
        """
        pred_t = self.model_t.predict(X)
        pred_c = self.model_c.predict(X)
        # The predictions now contain 'fmu' for both positive and
        # negative label. We are interested in the difference between
        # the positive labels at this point.
        pred_t = self.model_t.predict(X)
        pred_c = self.model_c.predict(X)
        tau = [item_t[1] - item_c[1] for item_t, item_c in zip(pred_t['mu'], pred_c['mu'])]
        return np.array(tau)            

    def generate_sample(self, X):
        """
        Bla?
        Focus on positive label?
        """
        pred_t = self.model_t.generate_sample(X)
        pred_c = self.model_c.generate_sample(X)
        # The predictions now contain 'fmu' for both positive and
        # negative label. We are interested in the difference between
        # the positive labels at this point.
        uplift_samples = [item_t - item_c for item_t, item_c in zip(pred_t, pred_c)]
        uplift_samples = np.array(uplift_samples)
        return uplift_samples

    def predict_uncertainty(self, X, mc_samples=1000):
        """
        Estimate uncertainty for _one_ observation.
        -Plot? Just parameters? Monte Carlo sampling?

        Args:
        X (np.array): Features of _one_ observation.
        mc_samples (int): Number of observations to draw in Monte
         Carlo sampling process.
        """
        tmp_t = self.model_t.predict_params(X.reshape(1, -1))
        tmp_c = self.model_c.predict_params(X.reshape(1, -1))
        samples_t = self.model_t.uncertainty_of_prediction(tmp_t['fmu'], tmp_t['fs2'], mc_samples)
        samples_c = self.model_c.uncertainty_of_prediction(tmp_c['fmu'], tmp_c['fs2'], mc_samples)
        tau_samples = [item_t[1] - item_c[1] for item_t, item_c in zip(samples_t, samples_c)]
        return tau_samples

    def get_credible_intervals(self, X, p_mass=0.95, mc_samples=1000):
        """
        Method for getting HPD credible intervals for all of X.

        Args:
        X (np.array): Features of observations
        p_mass (float): In ]0, 1]. Probability mass that needs to
         fall within credible interval.
        mc_samples (int): Size of sample to generate in MC.
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
            N = len(tau)
            n_interval = int(N * p_mass)
            lower_idx = 0
            upper_idx = n_interval
            # 3. Estimate width of sliding window
            smallest_width = np.inf
            for lower_idx in range(N - n_interval):
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
        tmp_t = self.model_t.predict_params(X)
        tmp_c = self.model_c.predict_params(X)
        intervals = []
        for i, _ in enumerate(X):
            # tmp = self.predict_uncertainty(row)  # Cannot use this here as it would slow down
            # code by 100x. Tried it.
            samples_t = self.model_t.uncertainty_of_prediction(tmp_t['fmu'][i], tmp_t['fs2'][i], mc_samples)
            samples_c = self.model_c.uncertainty_of_prediction(tmp_c['fmu'][i], tmp_c['fs2'][i], mc_samples)
            tmp = [item_t[1] - item_c[1] for item_t, item_c in zip(samples_t, samples_c)]
            interval = find_smallest_window(tmp, p_mass)
            #width_sum += find_smallest_window(tmp, p_mass)
            #width_sum += interval['width']
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
