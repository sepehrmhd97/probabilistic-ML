import numpy as np
from scipy.stats import norm
from gibbs import gibbs_sampler, gaussian_approximation
from moment_matching import run_moment_matching
from adf import run_adf, run_onestep_preds
from preprocessor import preprocess_dataset
from utils import plot_trace_means_stds, plot_hists, plot_priors_posteriors, output_ranking, randomize_csv, plot_gibbs_vs_moment_matching
import time

# Configurable Parameters:
# The mu, sigma, sigma_t values are set to the values used in the TrueSkill paper
mu = 25
sigma = 25 / 3
sigma_t = sigma / 2
burn_in = 3500

# Run the Gibbs sampler for Q4.a and produce the plots
def q4_a(n_iterations):

    # Run the Gibbs sampler
    s1_samples, s2_samples = gibbs_sampler(n_iterations, burn_in, mu, mu, sigma, sigma, sigma_t)

    # Plot the trace, means, and stds of the samples
    plot_trace_means_stds(s1_samples, s2_samples, display=False)

    # Run the Gibbs sampler again
    s1_samples, s2_samples = gibbs_sampler(n_iterations, burn_in, mu, mu, sigma, sigma, sigma_t)

    # Plot the figures again with burn-in
    plot_trace_means_stds(s1_samples, s2_samples, burn_in=burn_in, display=False)

    return s1_samples, s2_samples

# Approximate and return Gaussian dist. based on the mean and variance of the samples
def q4_b(s1_samples, s2_samples):
    dist_s1 = gaussian_approximation(s1_samples, burn_in)
    dist_s2 = gaussian_approximation(s2_samples, burn_in)

    return dist_s1, dist_s2

# Run Gibbs sampling with different numbers of iterations and plot the results
def q4_c():

    # Run the Gibbs sampler for different numbers of iterations
    for n_iters in [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000]:
        # Run the Gibbs sampler and time it
        start = time.time()
        s1_samples, s2_samples = gibbs_sampler(n_iters, burn_in, mu, mu, sigma, sigma, sigma_t)
        end = time.time()
        print("Time taken for {} iterations: {:3f} seconds".format(n_iters, end - start))
        
        # Plot the histograms and Gaussian approximations of the samples
        dist_s1 = gaussian_approximation(s1_samples, burn_in)
        dist_s2 = gaussian_approximation(s2_samples, burn_in)
        plot_hists(s1_samples, s2_samples, dist_s1, dist_s2, n_iters)


# Function to compare prior and posterior distributions after gibbs sampling
def q4_d(n_iters):

    # Run the Gibbs sampler
    s1_samples, s2_samples = gibbs_sampler(n_iters, burn_in, mu, mu, sigma, sigma, sigma_t)

    # Approximate the posterior distributions 
    post_s1 = gaussian_approximation(s1_samples, burn_in)
    post_s2 = gaussian_approximation(s2_samples, burn_in)
    # Approximate the prior distributions
    prior_s1 = norm(loc=mu, scale=sigma)
    prior_s2 = norm(loc=mu, scale=sigma)

    # Plot the prior and posterior distributions
    plot_priors_posteriors(post_s1, post_s2, prior_s1, prior_s2)

# Function that processes the data and updates the skill parameters for the teams for Q5
def q5(n_iters):

    # Run the ADF algorithm
    team_skills = run_adf("./dataset/SerieA.csv", n_iters, burn_in, mu, sigma, sigma_t)

    # Output the ranking of the teams to a file
    output_ranking(team_skills, "rankings.txt")

    # Randomize the order of the matches
    randomize_csv("./dataset/SerieA.csv", "./dataset/SerieA_randomized.csv")

    print("Randomized the order of the matches! Re-running ADF with randomized matches...", end="\n\n")

    # Run the ADF algorithm again
    team_skills = run_adf("./dataset/SerieA_randomized.csv", n_iters, burn_in, mu, sigma, sigma_t)

    # Output the ranking of the teams to a file
    output_ranking(team_skills, "rankings_randomized.txt")

# Function that runs one-step ahead predictions for Q6
def q6(n_iters):
    team_skills = run_onestep_preds("./dataset/SerieA.csv", n_iters, burn_in, mu, sigma, sigma_t)

# Function that runs moment matching for Q8
def q8(n_iters):
    # Run moment matching
    s1_m, s1_s, s2_m, s2_s = run_moment_matching(mu, sigma, sigma_t, 1)

    # Fit Gaussain distributions with the mean and variance from moment matching
    dist_s1 = norm(loc=s1_m, scale=s1_s)
    dist_s2 = norm(loc=s2_m, scale=s2_s)

    # Run the Gibbs sampler
    s1_samples, s2_samples = gibbs_sampler(n_iters, burn_in, mu, mu, sigma, sigma, sigma_t)

    # Plot the Gaussian approximations and the samples
    plot_gibbs_vs_moment_matching(s1_samples, s2_samples, dist_s1, dist_s2, n_iters, burn_in)

# Function that runs ADF on a new dataset for Q9
def q9(n_iters):

    # Preprocess the dataset and write it into a CSV file
    preprocess_dataset("./dataset/Halo2-HeadToHead.objml", "./dataset/Halo2-HeadToHead.csv", n_matches=350)

    #  Run the ADF algorithm
    team_skills = run_adf("./dataset/Halo2-HeadToHead.csv", n_iters, burn_in, mu, sigma, sigma_t)

    # Output to file the ranking of the players using a conservative metric
    output_ranking(team_skills, "rankings_halo.txt", conservative=True)



if __name__ == "__main__":

    # Q4.a
    print("Running Gibbs sampler for Q4.a")
    s1_samples, s2_samples = q4_a(20000)
    print("Finished running Gibbs sampler for Q4.a!", end="\n\n")
    # Q4.b
    print("Fitting Gaussians for Q4.b")
    dist_s1, dist_s2 = q4_b(s1_samples, s2_samples)
    print("Finished fitting Gaussians for Q4.b!", end="\n\n")
    # Q4.c
    print("Running Gibbs sampler for different n_iterations for Q4.c")
    q4_c()
    print("Finished running Gibbs sampler for different iterations for Q4.c!", end="\n\n")
    # Q4.d
    print("Plotting the posterior and prior distributions of s1 and s2 for Q4.d")
    q4_d(20000)
    print("Finished plotting the posterior and prior distributions of s1 and s2 for Q4.d!", end="\n\n")

    # Q5
    print("Running ADF for Q5")
    q5(20000)
    print("Finished running ADF for Q5!", end="\n\n")

    # Q6 
    print("Running One-step ahead predictions for Q6")
    q6(20000)
    print("Finished running One-step ahead predictions for Q6!", end="\n\n")

    # Q8
    print("Running moment matching and comparing to Gibbs sampler for Q8")
    q8(20000)
    print("Finished running moment matching and comparing to Gibbs sampler for Q8!", end="\n\n")

    # Q9
    print("Running ADF on a new dataset for Q9")
    q9(20000)
    print("Finished running ADF on a new dataset for Q9!", end="\n\n")
