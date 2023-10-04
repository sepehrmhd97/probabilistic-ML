import numpy as np
from scipy.stats import truncnorm

# The multiply, divide and truncGaussMM functions are taken from the solution to exercise 7.2

def mutiplyGauss(m1, s1, m2, s2):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1 ,s1)*N(m2 ,s2)
    #
    # Input:
    # m1, s1: mean and variance of first Gaussian
    # m2, s2: mean and variance of second Gaussian
    #
    # Output:
    # m, s: mean and variance of the product Gaussian

    s = 1 / (1 / s1 + 1 / s2)
    m = (m1 / s1 + m2 / s2) * s
    return m, s

def divideGauss(m1, s1, m2, s2):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1 ,s1)/N(m2 ,s2)
    #
    # Input:
    # m1, s1: mean and variance of the numerator Gaussian
    # m2, s2: mean and variance of the denominator Gaussian
    #
    # Output:
    # m, s: mean and variance of the quotient Gaussian

    m, s = mutiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):
    # computes the mean and variance of a truncated Gaussian distribution
    #
    # Input:
    # a, b: The interval [a, b] on which the Gaussian is being truncated
    # m0, s0: mean and variance of the Gaussian which is to be truncated
    #
    # Output:
    # m, s: mean and variance of the truncated Gaussian

    # scale interval with mean and variance
    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s


def run_moment_matching(mu, sigma, sigma_t, y):
    m0_s1 = m0_s2 = mu # The means of the priors p(s1) and p(s2)
    s0_s1 = s0_s2 = sigma**2 # The variances of the priors p(s1) and p(s2)
    st0 = sigma_t**2 # The variance of the prior p(t)
    y0 = y # The measurement of y

    # Message mu1 from factor f_s1 to node s1
    mu1_m = m0_s1 # mean of message
    mu1_s = s0_s1 # variance of message

    # Message mu2 from factor f_s2 to node s2
    mu2_m = m0_s2 # mean of message
    mu2_s = s0_s2 # variance of message

    # Message mu4 from variable s1 to factor f_t_s1_s2
    mu4_m = mu1_m # mean of message
    mu4_s = mu1_s # variance of message

    # Message mu5 from variable s2 to factor f_t_s1_s2
    mu5_m = mu2_m # mean of message
    mu5_s = mu2_s # variance of message

    # Message mu6 from factor f_t_s1_s2 to variable t
    mu6_m = mu4_m - mu5_m # mean of message
    mu6_s = mu4_s + mu5_s + st0 # variance of message

    # Do moment matching of the marginal of t
    if y0 == 1:
        a, b = 0, np.Inf
    else:
        a, b = np.NINF, 0

    pt_m, pt_s = truncGaussMM(a, b, mu6_m, mu6_s)

    # Compute the message from t to f_t_s1_s2
    mu8_m, mu8_s = divideGauss(pt_m, pt_s, mu6_m, mu6_s)

    # Compute the message from f_t_s1_s2 to s1
    mu10_m = mu8_m + mu5_m # mean of message
    mu10_s = mu8_s + mu5_s + st0 # variance of message

    # Compute the posterior of s1
    ps1_m, ps1_s = mutiplyGauss(m0_s1, s0_s1, mu10_m, mu10_s)

    # Compute the message from f_t_s1_s2 to s2
    mu9_m = mu4_m - mu8_m # mean of message TODO: Check if this is correct
    mu9_s = mu4_s + mu8_s + st0 # variance of message

    # Compute the posterior of s2
    ps2_m, ps2_s = mutiplyGauss(m0_s2, s0_s2, mu9_m, mu9_s)

    return ps1_m, np.sqrt(ps1_s), ps2_m, np.sqrt(ps2_s)
