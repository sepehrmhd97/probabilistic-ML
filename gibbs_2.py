import numpy as np
from scipy.stats import truncnorm, multivariate_normal
from scipy.linalg import inv
import matplotlib.pyplot as plt

# Gibbs sampler for to estimate the posterior distribution of p(s1, s2 | y) where y = 1
def gibbs_sampler(n_iterations, burn_in, mu, sigma, sigma_t):

    s1_samples = np.zeros(n_iterations + burn_in)
    s2_samples = np.zeros(n_iterations + burn_in)

    s1_samples[0] = mu
    s2_samples[0] = mu

    for i in range(1, burn_in + n_iterations):

        # Sample T first from p(t | s1, s2, y=1) => Q3.b
        mu_t = s1_samples[i-1] - s2_samples[i-1]
        a_t, b_t = (0 - mu_t) / sigma_t, np.inf
        t = truncnorm.rvs(a_t, b_t, loc=mu_t, scale=sigma_t)

        # Sample s1 and s2 from p(s1, s2 | t, y=1) => Q3.a
        # term_1 = np.outer(np.array([1, -1]), np.array([1, -1])) * 1/(sigma_t**2)
        # term_2 = np.array([[sigma**2, 0], [0, sigma**2]]) * 1/(sigma**2 * sigma**2)
        # cov = inv(term_1 + term_2)

        # term_3 = 1/(sigma**2 * sigma**2) * np.matmul(np.array([[sigma**2, 0], [0, sigma**2]]), np.array([mu, mu]))
        # term_4 = 1/(sigma_t**2) * t * np.array([1, -1])
        # mean = np.matmul(cov, term_3 + term_4)

        S0 = np.mat([[sigma ** 2, 0], [0, sigma ** 2]])
        B = 1 / sigma ** 2
        X = np.mat([1, -1])
        m0 = np.mat([mu, mu]).T
        y = t
        SN_inverse = np.linalg.inv(S0) + B * X.T * X
        SN = np.linalg.inv(SN_inverse)
        m = SN * (np.linalg.inv(S0) * m0 + B * X.T * y)

        s1_samples[i], s2_samples[i] = multivariate_normal.rvs(mean=np.asarray(m).squeeze(), cov=SN, size=1, random_state=None)

    return s1_samples, s2_samples

# Parameters
n_iterations = 5000
burn_in = 10
mu = 25
sigma = 8.3
sigma_t = 3.3
init_vals = (-1, 1)

s1_samples, s2_samples = gibbs_sampler(n_iterations, burn_in, mu, sigma, sigma_t)

plt.figure(figsize=(12, 6))

# Trace for s1
plt.subplot(1, 2, 1)
plt.plot(np.arange(n_iterations + burn_in), s1_samples, label="s1 samples", alpha=0.6)
# plt.axvline(x=burn_in, color='r', linestyle='--', label="Burn-in")
plt.title("Trace plot for s1")
plt.xlabel("Iterations")
plt.ylabel("s1")
plt.legend()

# Trace for s2
plt.subplot(1, 2, 2)
plt.plot(np.arange(n_iterations + burn_in), s2_samples, label="s2 samples", alpha=0.6)
# plt.axvline(x=burn_in, color='r', linestyle='--', label="Burn-in")
plt.title("Trace plot for s2")
plt.xlabel("Iterations")
plt.ylabel("s2")
plt.legend()

plt.show()

# sigma_1 = 1
# sigma_2 = 4
# sigma_t = 5
# mu_1 = 1
# mu_2 = -1
# t = 3
# # Multiply a column vector to a row one to get a matrix
# term_1 = np.outer(np.array([1, -1]), np.array([1, -1])) * 1/(sigma_t)
# term_2 = np.array([[sigma_2, 0], [0, sigma_1]]) * 1/(sigma_1 * sigma_2)
# cov = inv(term_1 + term_2)

# term_3 = 1/(sigma_1*sigma_2) * np.matmul(np.array([[sigma_2, 0], [0, sigma_1]]), np.array([mu_1, mu_2]))
# term_4 = 1/(sigma_t) * t * np.array([1, -1])

# mean = np.matmul(cov, term_3 + term_4)

