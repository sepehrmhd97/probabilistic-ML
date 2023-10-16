import numpy as np
from scipy.stats import truncnorm, multivariate_normal, norm
from scipy.linalg import inv

# Gibbs sampler for to estimate the posterior distribution of p(s1, s2 | y) 
def gibbs_sampler(n_iterations, burn_in, mu_1, mu_2, sigma_1, sigma_2, sigma_t, y=1, s1_update_coeff=1, s2_update_coeff=1):

    s1_samples = np.zeros(n_iterations + burn_in)
    s2_samples = np.zeros(n_iterations + burn_in)

    s1_samples[0] = mu_1
    s2_samples[0] = mu_2

    for i in range(1, burn_in + n_iterations):

        # Sample T first from p(t | s1, s2, y) => Q3.b

        # Mu_t = s1 - s2
        mu_t = s1_samples[i-1] - s2_samples[i-1]


        # If team1 has won
        if (y == 1):
            # Sample from truncated normal distribution
            # With lower bound = 0 & upper bound = infinity
            t = truncnorm.rvs((0 - mu_t) / sigma_t, np.inf, loc=mu_t, scale=sigma_t)
        # If team2 has won
        elif (y == -1):
            # Sample from truncated normal distribution
            # With lower bound = -infinity & upper bound = 0
            print(sigma_t)
            t = truncnorm.rvs(-np.inf, (0 - mu_t) / sigma_t, loc=mu_t, scale=sigma_t)
        # If the match is a draw
        else:
            # Set t to zero in order to not favor any team
            t = 0

        # # Set the A vector as the team1 and 2 update coefficients
        A = np.array([s1_update_coeff, -s2_update_coeff])

        # # calculate the magnitude of the vector [s1_update_coeff, s2_update_coeff]
        # magnitude = np.sqrt(s1_update_coeff**2 + s2_update_coeff**2)

        # # normalize the coefficients
        # s1_update_coeff_normalized = s1_update_coeff / magnitude
        # s2_update_coeff_normalized = s2_update_coeff / magnitude

        # # print(s1_update_coeff_normalized, s2_update_coeff_normalized)

        # # create the array A with the normalized coefficients
        # A = np.array([1, -1])

        # Sample s1 and s2 from p(s1, s2 | t, y) => Q3.a

        # Calculate the covariance of the multivariate normal distribution
        term_1 = np.outer(A, A) * 1/(sigma_t**2)
        term_2 = np.array([[sigma_2**2, 0], [0, sigma_1**2]]) * 1/(sigma_1**2 * sigma_2**2)
        cov = inv(term_1 + term_2)

        # print(cov)
        # print(sigma_t)

        # Calculate the mean of the multivariate normal distribution
        term_3 = 1/(sigma_2**2 * sigma_1**2) * np.matmul(np.array([[sigma_2**2, 0], [0, sigma_1**2]]), np.array([mu_1, mu_2]))

        term_4 = 1/(sigma_t**2) * t * A

        # max_value = 1e10  # or whatever you deem appropriate
        # if np.abs(term_4) > max_value:
        #     term_4 = np.sign(term_4) * max_value

        print(term_4)

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
