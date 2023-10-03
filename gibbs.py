import numpy as np
from scipy.stats import truncnorm, multivariate_normal, norm
from scipy.linalg import inv

# Gibbs sampler for to estimate the posterior distribution of p(s1, s2 | y) 
def gibbs_sampler(n_iterations, burn_in, mu_1, mu_2, sigma_1, sigma_2, sigma_t, y=1):

    s1_samples = np.zeros(n_iterations + burn_in)
    s2_samples = np.zeros(n_iterations + burn_in)

    s1_samples[0] = mu_1
    s2_samples[0] = mu_2

    for i in range(1, burn_in + n_iterations):

        # Sample T first from p(t | s1, s2, y) => Q3.b

        # Mu_t = s1 - s2
        mu_t = s1_samples[i-1] - s2_samples[i-1]
        if (y == 1):
            # Sample from truncated normal distribution
            # With lower bound = 0 & upper bound = infinity
            t = truncnorm.rvs((0 - mu_t) / sigma_t, np.inf, loc=mu_t, scale=sigma_t)
        else:
            # Sample from truncated normal distribution
            # With lower bound = -infinity & upper bound = 0
            t = truncnorm.rvs(-np.inf, (0 - mu_t) / sigma_t, loc=mu_t, scale=sigma_t)

        # Sample s1 and s2 from p(s1, s2 | t, y) => Q3.a

        # Calculate the covariance of the multivariate normal distribution
        term_1 = np.outer(np.array([1, -1]), np.array([1, -1])) * 1/(sigma_t**2)
        term_2 = np.array([[sigma_2**2, 0], [0, sigma_1**2]]) * 1/(sigma_1**2 * sigma_2**2)
        cov = inv(term_1 + term_2)

        # Calculate the mean of the multivariate normal distribution
        term_3 = 1/(sigma_2**2 * sigma_1**2) * np.matmul(np.array([[sigma_2**2, 0], [0, sigma_1**2]]), np.array([mu_1, mu_2]))
        term_4 = 1/(sigma_t**2) * t * np.array([1, -1])
        mean = np.matmul(cov, term_3 + term_4)

        # Sample from the multivariate normal distribution
        s1_samples[i], s2_samples[i] = multivariate_normal.rvs(mean=mean, cov=cov, size=1, random_state=None)

    return s1_samples, s2_samples

# Function that finds a Gaussian approximation of the posterior distribution based on the samples
def gaussian_approximation(samples, burn_in):

    # Calculate the mean and std of the samples
    mean = np.mean(samples[burn_in:])
    std = np.std(samples[burn_in:])

    dist = norm(loc=mean, scale=std)

    return dist
