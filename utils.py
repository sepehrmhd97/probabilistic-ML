import matplotlib.pyplot as plt
import numpy as np

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
    plt.subplot(1, 2, 1)
    plt.plot(s1_samples, label="s1 samples", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Trace plot for s1")
    plt.xlabel("Iterations")
    plt.ylabel("s1")
    plt.legend()

    # Trace for s2
    plt.subplot(1, 2, 2)
    plt.plot(s2_samples, label="s2 samples", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Trace plot for s2")
    plt.xlabel("Iterations")
    plt.ylabel("s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_trace_burnin.png')
    else:
        plt.savefig('gibbs_trace.png')

    # Plotting the mean of the samples throughout the sampling process

    plt.figure(figsize=(12, 6))

    # Means of s1
    plt.subplot(1, 2, 1)
    plt.plot(s1_means, label="s1 mean", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Means of s1")
    plt.xlabel("Iterations")
    plt.ylabel("s1")
    plt.legend()

    # Means of s2
    plt.subplot(1, 2, 2)
    plt.plot(s2_means, label="s2 mean", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Means of s2")
    plt.xlabel("Iterations")
    plt.ylabel("s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_means_burnin.png')
    else:
        plt.savefig('gibbs_means.png')

    # Plotting the std of the samples throughout the sampling process

    plt.figure(figsize=(12, 6))

    # Means of s1
    plt.subplot(1, 2, 1)
    plt.plot(s1_sigmas, label="s1 std", color='blue')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Standard deviations of s1")
    plt.xlabel("Iterations")
    plt.ylabel("s1")
    plt.legend()

    # Means of s2
    plt.subplot(1, 2, 2)
    plt.plot(s2_sigmas, label="s2 std", color='orange')
    if burn_in is not None:
        plt.axvline(x=burn_in, color='red', linestyle='--')
    plt.title("Standard deviations of s2")
    plt.xlabel("Iterations")
    plt.ylabel("s2")
    plt.legend()

    if burn_in is not None:
        plt.savefig('gibbs_stds_burnin.png')
    else:
        plt.savefig('gibbs_stds.png')

    if display:
        plt.show()
    else:
        plt.close('all')