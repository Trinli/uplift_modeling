"""
This file will contain the Monte Carlo (MC) approaches to
estimate uncertainty from a beta-difference distribution
(Pham-Gia, Turkkan & Eng, 1993).

1. Estimation of credible interval width
 -PAVA for all bins to the left of \hat{tau}
  and separately for all the bins to the right
  of \hat{tau}?
2. Plot the beta-difference distribution
3. Statistical test for how likely tau_1
 is larger than tau_2 (p(tau_1 > tau_2)).

"""
import numpy as np
from scipy.stats import beta
#import matplotlib.pyplot as plt


def uncertainty(alpha1, beta1, alpha0, beta0,
                prior_a1=1, prior_b1=1,
                prior_a0=1, prior_b0=1,
                N=100000,
                p_mass=0.95):
    """
    Function for estimating highest posterior density
    credible intervals (HPD-CI).
    Distribution 1 is assumed to be p(y=1|x, do(t=1)) and
    distribution 0 p(y=1|x, do(t=0)).

    Args:
    alpha1: Alpha for distribution 1
    beta1: Beta for distribution 1
    etc.
    prior_a1: Alpha prior for distribution 1
    etc.
    N: Number of samples to draw. Should probably be
     at least 10,000.
    bins: 2 / bins is the resolution of the distribution.
    p (float): In [0, 1]. The width in terms of probability
     mass for the credible interval desired.
    """
    # Draw samples from distribution
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


def test_for_difference(alpha11, beta11, alpha12, beta12,
                        alpha21, beta21, alpha22, beta22,
                        prior_a11=1, prior_b11=1,
                        prior_a12=1, prior_b12=1,
                        prior_a21=1, prior_b21=1,
                        prior_a22=1, prior_b22=1,
                        N=100000):
    """
    Bayesian test for difference between two uplift estimates.

    Args:
    alpha11: Alpha for p(y=1|x, t=1) for tau_1
    beta11: Beta for p(y=1|x, t=1) for tau_1
    etc.
    prior_a21: Alpha prior for p(y=1|x, t=1) for tau_2
    etc.
    N: Number of samples to draw. Should probably be
     at least 10,000.
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
