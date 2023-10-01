
import numpy as np
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt

# Gibbs sampler for to estimate the posterior distribution of p(s1, s2 | y) where y = 1
def gibbs_sampler(n_iterations=10000, burn_in=1000, mu=0, sigma=1, init_vals=(0, 0)):

    s1_samples = np.zeros(n_iterations + burn_in)
    s2_samples = np.zeros(n_iterations + burn_in)

    s1_samples[0] = init_vals[0]
    s2_samples[0] = init_vals[1]

    for i in range(1, burn_in + n_iterations):

        # Computing the bounds of the truncated normal distributions

        a1, b1 = (s2_samples[i-1] - mu) / sigma, np.inf
        mu = # ... s2 ...
        s1_samples[i] = truncnorm.rvs(a1, b1, loc=mu, scale=sigma)

        # Sample s1 and s2 from p(s1, s2 | y=1)
        a2, b2 = -np.inf, (s1_samples[i] - mu) / sigma
        s2_samples[i] = truncnorm.rvs(a2, b2, loc=mu, scale=sigma)

    return s1_samples, s2_samples

# Parameters
n_iterations = 3
burn_in = 1000
mu = 0
sigma = 1
init_vals = (-10, 10)

s1_samples, s2_samples = gibbs_sampler(n_iterations=n_iterations, burn_in=burn_in, mu=mu, sigma=sigma, init_vals=init_vals)

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
