import numpy as np
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt

def gibbs_sampler(iterations, burn_in, mu, sigma):
    s1_samples = np.zeros(iterations + burn_in)
    s2_samples = np.zeros(iterations + burn_in)
    
    # Initial values
    s1_samples[0], s2_samples[0] = mu, mu  # Starting from the prior mean
    
    for i in range(1, iterations + burn_in):
        # Sample s1 and s2 from p(s1, s2|y, t)
        a1, b1 = (mu - s2_samples[i-1]) / sigma, np.inf
        a2, b2 = -np.inf, (mu - s1_samples[i-1]) / sigma
        
        s1_samples[i] = truncnorm.rvs(a1, b1, loc=mu, scale=sigma)
        s2_samples[i] = truncnorm.rvs(a2, b2, loc=mu, scale=sigma)
    
    return s1_samples, s2_samples

# Parameters
iterations = 1000
burn_in = 100
mu = 0  # Prior mean
sigma = 1  # Prior standard deviation

s1_samples, s2_samples = gibbs_sampler(iterations, burn_in, mu, sigma)

print(s1_samples)


plt.figure(figsize=(12, 6))

# Trace for s1
plt.subplot(1, 2, 1)
plt.plot(np.arange(iterations + burn_in), s1_samples, label="s1 samples", alpha=0.6)
plt.axvline(x=burn_in, color='r', linestyle='--', label="Burn-in")
plt.title("Trace plot for s1")
plt.xlabel("Iterations")
plt.ylabel("s1")
plt.legend()

# Trace for s2
plt.subplot(1, 2, 2)
plt.plot(np.arange(iterations + burn_in), s2_samples, label="s2 samples", alpha=0.6)
plt.axvline(x=burn_in, color='r', linestyle='--', label="Burn-in")
plt.title("Trace plot for s2")
plt.xlabel("Iterations")
plt.ylabel("s2")
plt.legend()

plt.tight_layout()
plt.show()

# Histograms with Gaussian approximation
# ... [previous histogram code] ...



# # Estimate the mean and covariance of the samples
# mean_s1 = np.mean(s1_samples)
# mean_s2 = np.mean(s2_samples)
# cov_matrix = np.cov(s1_samples, s2_samples)

# print("Estimated mean for s1:", mean_s1)
# print("Estimated mean for s2:", mean_s2)
# print("Estimated covariance matrix:")
# print(cov_matrix)

# # Plot histograms of the samples with the Gaussian approximation
# x = np.linspace(min(s1_samples), max(s1_samples), 1000)
# y = np.linspace(min(s2_samples), max(s2_samples), 1000)
# X, Y = np.meshgrid(x, y)
# Z = np.exp(-0.5 * (cov_matrix[0, 0] * (X - mean_s1)**2 + 
#                   2 * cov_matrix[0, 1] * (X - mean_s1) * (Y - mean_s2) + 
#                   cov_matrix[1, 1] * (Y - mean_s2)**2))

# plt.figure(figsize=(12, 6))

# # Histogram for s1
# plt.subplot(1, 2, 1)
# plt.hist(s1_samples, bins=50, density=True, alpha=0.6, label="Samples")
# plt.plot(x, np.exp(-(x - mean_s1)**2 / (2 * cov_matrix[0, 0])) / np.sqrt(2 * np.pi * cov_matrix[0, 0]), 
#          label="Gaussian Approximation")
# plt.title("Posterior distribution for s1")
# plt.legend()

# # Histogram for s2
# plt.subplot(1, 2, 2)
# plt.hist(s2_samples, bins=50, density=True, alpha=0.6, label="Samples")
# plt.plot(y, np.exp(-(y - mean_s2)**2 / (2 * cov_matrix[1, 1])) / np.sqrt(2 * np.pi * cov_matrix[1, 1]), 
#          label="Gaussian Approximation")
# plt.title("Posterior distribution for s2")
# plt.legend()

# plt.tight_layout()
# plt.show()
