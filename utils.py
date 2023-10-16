import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random

# Set the font properties
font = {'family': 'Times New Roman', 'size': 15}
plt.rc('font', **font)

# Function that plots the trace, means, and stds of the samples throughout the sampling process
def plot_trace_means_stds(s1_samples, s2_samples, burn_in = None, display=True):

    # Calculate the mean and sigma of the samples in different iterations
    n_samples = len(s1_samples)
    s1_means, s2_means = np.zeros(n_samples), np.zeros(n_samples)
    s1_sigmas, s2_sigmas = np.zeros(n_samples), np.zeros(n_samples)
    for i in range(1, n_samples):
        s1_means[i] = np.mean(s1_samples[0:i])
        s2_means[i] = np.mean(s2_samples[0:i])
        s1_sigmas[i] = np.std(s1_samples[0:i])
        s2_sigmas[i] = np.std(s2_samples[0:i])

    # Plotting the trace plots of the samples
    plt.figure(figsize=(12, 6))

    # Trace for s1
    plt.plot(s1_samples, label="s1 samples", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')

    # Trace for s2
    plt.plot(s2_samples, label="s2 samples", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Trace plots for s1 and s2")
    plt.xlabel("Iterations")
    plt.ylabel("s1, s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_trace_burnin.svg', bbox_inches="tight")
    else:
        plt.savefig('gibbs_trace.svg', bbox_inches="tight")

    # Plotting the mean of the samples throughout the sampling process

    plt.figure(figsize=(12, 6))

    # Means of s1
    plt.plot(s1_means, label="s1 mean", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')

    # Means of s2
    plt.plot(s2_means, label="s2 mean", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Means of s1 and s2")
    plt.xlabel("Iterations")
    plt.ylabel("s1, s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_means_burnin.svg', bbox_inches="tight")
    else:
        plt.savefig('gibbs_means.svg', bbox_inches="tight")

    # Plotting the std of the samples throughout the sampling process

    plt.figure(figsize=(12, 6))

    # Means of s1
    plt.plot(s1_sigmas, label="s1 std", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')

    # Means of s2
    plt.plot(s2_sigmas, label="s2 std", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Standard deviations of s1 and s2")
    plt.xlabel("Iterations")
    plt.ylabel("s1, s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_stds_burnin.svg', bbox_inches="tight")
    else:
        plt.savefig('gibbs_stds.svg', bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close('all')

# Function that plots the histograms of the samples with the Gaussian approximations
def plot_hists(s1_samples, s2_samples, dist_s1, dist_s2, n_iterations):

    # Plotting the histograms of the samples and the Gaussian approximations in the same plot
    plt.figure(figsize=(12, 6))

    # Histogram of s1
    plt.hist(s1_samples, bins=50, density=True, label="s1 samples", color='blue')
    plt.plot(np.linspace(0, 50, 100), dist_s1.pdf(np.linspace(0, 50, 100)), label="s1 Gaussian approximation", color='red')

    # Histogram of s2
    plt.hist(s2_samples, bins=50, density=True, label="s2 samples", color='orange')
    plt.plot(np.linspace(0, 50, 100), dist_s2.pdf(np.linspace(0, 50, 100)), label="s2 Gaussian approximation", color='green')
    plt.title("Histogram and Gaussian approx. of s1 and s2")
    plt.xlabel("s1, s2")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f'gibbs_hists_{n_iterations}.svg', bbox_inches="tight")

# Function that plots posterior distributions of s1 and s2 with the prior distributions
def plot_priors_posteriors(post_s1, post_s2, prior_s1, prior_s2):

    # Plot the posterior distributions of s1 and s2 with the prior distributions
    plt.figure(figsize=(12, 6))

    # Posterior of s1
    plt.plot(np.linspace(0, 50, 100), post_s1.pdf(np.linspace(0, 50, 100)), label="posterior of s1", color='blue')
    plt.plot(np.linspace(0, 50, 100), prior_s1.pdf(np.linspace(0, 50, 100)), label="prior of s1 and s2", color='red')

    # Posterior of s2
    plt.plot(np.linspace(0, 50, 100), post_s2.pdf(np.linspace(0, 50, 100)), label="posterior of s2", color='orange')
    plt.title("Posterior and prior distributions of s1 and s2")
    plt.xlabel("s1, s2")
    plt.ylabel("Probability")
    plt.legend()

    plt.savefig('gibbs_priors_posteriors.svg', bbox_inches="tight")

# Function that outputs the ranking of the teams based on their skill parameters to a file
def output_ranking(team_skills, filename, conservative=False):

    # Sort the teams based on their mean skill parameters or conservative (mean - 3*std) skill parameters
    if conservative:
        sorted_teams = sorted(team_skills.items(), key=lambda x: x[1][0] - 3*x[1][1], reverse=True)
    else:
        sorted_teams = sorted(team_skills.items(), key=lambda x: x[1][0], reverse=True)

    # Write the sorted teams to a file with their rank, name, mean, and std
    with open(filename, "w") as f:
        for i, (team, (mean, std)) in enumerate(sorted_teams):
            f.write("{},{},{},{}\n".format(i+1, team, mean, std))
    
    return sorted_teams

# Function that randomizes the rows of a CSV file
def randomize_csv(filename, new_filename):

    # Read the CSV file
    with open(filename, "r") as f:
        lines = f.readlines()

    # Randomize the lines except the first line
    random.shuffle(lines[1:])

    # Write the randomized lines to a new CSV file
    with open(new_filename, "w") as f:
        f.writelines(lines)

# Function that plots the samples of a Gibbs sampler alongside the Gaussian approximations from the samples and moment matching
def plot_gibbs_vs_moment_matching(s1_samples, s2_samples, dist_s1, dist_s2, n_iterations, burn_in):

    # Fit Gaussian distributions to the samples
    dist_s1_samples = norm.fit(s1_samples[burn_in:])
    dist_s2_samples = norm.fit(s2_samples[burn_in:])

    # Print moment matching means and stds
    print("Moment matching means and stds:")
    print("s1: ", dist_s1.mean(), dist_s1.std())
    print("s2: ", dist_s2.mean(), dist_s2.std())

    # Print means and stds from Gibbs samples
    print("Gibbs sampler means and stds:")
    print("s1: ", dist_s1_samples[0], dist_s1_samples[1])
    print("s2: ", dist_s2_samples[0], dist_s2_samples[1])
    
    # Plotting the histograms of the samples and the Gaussian approximations in the same plot
    plt.figure(figsize=(12, 6))

    # Histogram of s1
    plt.hist(s1_samples, bins=50, density=True, label="s1 samples", color='blue')
    plt.plot(np.linspace(0, 50, 100), norm.pdf(np.linspace(0, 50, 100), dist_s1_samples[0], dist_s1_samples[1]), label="s1 Gaussian (samples)", color='green')
    plt.plot(np.linspace(0, 50, 100), dist_s1.pdf(np.linspace(0, 50, 100)), label="s1 Gaussian (moment-matching)", color='red')

    # Histogram of s2
    plt.hist(s2_samples, bins=50, density=True, label="s2 samples", color='orange')
    plt.plot(np.linspace(0, 50, 100), norm.pdf(np.linspace(0, 50, 100), dist_s2_samples[0], dist_s2_samples[1]), label="s2 Gaussian (samples)", color='green')
    plt.plot(np.linspace(0, 50, 100), dist_s2.pdf(np.linspace(0, 50, 100)), label="s2 Gaussian (moment-matching)", color='red')
    plt.title("Histogram and Gaussian approx. of s1 & s2 (Gibbs vs. moment matching)")
    plt.xlabel("s1, s2")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f'gibbs_vs_moment_matching.svg', bbox_inches="tight")


