import numpy as np
from gibbs import gibbs_sampler
from utils import plot_trace_means_stds

# Run the Gibbs sampler for Q4.a and produce the plots
def q4_a():
    # Parameters:
    # The mu, sigma, sigma_t values are set to the values used in the TrueSkill paper
    n_iterations = 20000
    burn_in = 3500
    mu = 25
    sigma = 25 / 3
    sigma_t = sigma / 2

    # Run the Gibbs sampler
    s1_samples, s2_samples = gibbs_sampler(n_iterations, burn_in, mu, sigma, sigma_t)

    # Plot the trace, means, and stds of the samples
    plot_trace_means_stds(s1_samples, s2_samples, display=False)

    # Run the Gibbs sampler again
    s1_samples, s2_samples = gibbs_sampler(n_iterations, burn_in, mu, sigma, sigma_t)

    # Plot the figures again with burn-in
    plot_trace_means_stds(s1_samples, s2_samples, burn_in=burn_in, display=True)


if __name__ == "__main__":
    q4_a()