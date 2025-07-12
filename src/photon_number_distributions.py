import numpy as np
from scipy.special import factorial
from numpy.polynomial.hermite import hermval

from src.utils import DGBSUtils

'''Methods to calculate total photon number distribution of Displaced GBS'''

def single_displaced_squeezed_vacuum(r, beta, cutoff):
    """
    Find the photon number distribution of a single mode displaced squeezed vacuum state up to cutoff
    """

    theta = np.angle(r)
    r = np.abs(r)

    x = (np.conj(beta) * np.exp(1j * theta / 2) * np.cosh(r) - beta * np.exp(-1j * theta / 2) * np.sinh(r)) \
        / (-1j * np.sqrt(np.sinh(2 * r)))
    c = np.diag(np.ones(cutoff+1, dtype=int))
    hermite2s = np.abs(hermval(x, c))**2  # The hermite mod squared values from n=0 to n=cutoff

    p_n = np.zeros(cutoff+1, dtype=float)
    for n in range(cutoff+1):
        p_n[n] = np.exp(- np.abs(beta)**2 + np.real(beta**2 * np.exp(-1j*theta)) * np.tanh(r)) * \
            (np.tanh(r) ** n) / (factorial(n) * (2 ** n) * np.cosh(r))

    p_n = p_n * hermite2s

    return p_n

def total_displaced_squeezed_vacuum(rs, betas, cutoff):
    """
    Find the total photon number distribution of K single-mode displaced squeezed vacuua up to total cutoff.
    If rs and betas has length 1, then calculate distribution on a single mode
    """
    rs = np.atleast_1d(rs)
    betas = np.atleast_1d(betas)

    if len(rs) != len(betas):
        raise ValueError('Input dimensions dont match')

    K = len(rs)

    if np.all(rs - rs[0] == 0) and np.all(betas - betas[0] == 0):
        # identical single mode states
        r = rs[0]
        beta = betas[0]

        p0 = single_displaced_squeezed_vacuum(r, beta, cutoff)
        p_n = np.zeros(cutoff+1)
        p_n[0] = 1
        for iter in range(K):
            p_n = np.convolve(p_n, p0)[:cutoff+1]

    else:
        # unidentical states
        p_n = np.ones(cutoff+1)
        p_n[0] = 1
        for iter in range(K):
            r = rs[iter]
            beta = betas[iter]

            p = single_displaced_squeezed_vacuum(r, beta, cutoff)
            p_n = np.convolve(p_n, p)[:cutoff+1]


    return p_n


def vac_prob(cov, means):
    """
    Vacuum probability for a general Gaussian state of M-modes
    :param cov: 2M*2M covariance matrix, both quadrature and Fock basis are allowed
    :param means: 2M means vector, both quadrature and Fock basis are allowed
    :return: The probability to measuring 0 photons
    """

    (n,m) = cov.shape
    k = len(means)
    if n != m:
        raise ValueError('Input covariance matrix should be square')
    if n % 2 != 0:
        raise ValueError('Input covariance matrix should have even dimensions')
    if n != k:
        raise ValueError('Input covariance matrix and means vector should have matching dimensions')
    M = n
    cov_Q = cov + np.identity(2 * M) / 2
    cov_Q_inv = np.linalg.inv(cov_Q)

    return np.exp(-0.5 * means.T @ cov_Q_inv @ means) / np.sqrt(np.linalg.det(cov_Q))

def vac_prob_displaced_squeezed_vacuum(rs, betas):
    """
    Vacuum probability for K single mode displaced squeezed vacuum states
    :param rs: (K,) squeezing parameters
    :param betas: (K,) displacement parameters
    :return: The probability of measuring 0 photons
    """

    rs = np.atleast_1d(rs)
    betas = np.atleast_1d(betas)

    if len(rs) != len(betas):
        raise ValueError('Input dimensions dont match')

    exponent = 2 * np.sum(np.abs(betas) ** 2) - np.sum( np.tanh(rs) * (betas ** 2 + np.conj(betas) ** 2) )

    return np.exp(-0.5 * exponent) / np.prod(np.absolute(np.cosh(rs)))


def big_F(w, K, cutoff):
    """
    This is the convolution of f(n) = |Hn(i*w/sqrt(2))|^2 / n!, over (n_1, ... n_K) where n_1+...+n_K = cutoff.
    Hn is the n-th Hermite polynomial
    """

    x = w * 1j / np.sqrt(2)
    c = np.diag(np.ones(cutoff + 1, dtype=int))
    hermite2s = np.abs(hermval(x, c)) ** 2  # The hermite mod squared values from n=0 to n=cutoff
    fn = hermite2s / factorial(np.arange(cutoff+1), exact=True)

    bigF = np.zeros(cutoff + 1)
    bigF[0] = 1
    for iter in range(K):
        bigF = np.convolve(bigF, fn)[:cutoff + 1]

    return bigF


# # Tests
# K = 5
# N_mean = 1
# N_cutoff = 10
# w = 1
# r, beta = DGBSUtils.solve_w(w, N_mean)
#
# rs = r * np.ones(K, dtype=float)
# betas = beta * np.ones(K, dtype=float)
#
# p0 = vac_prob_displaced_squeezed_vacuum(rs, betas)
# pN = total_displaced_squeezed_vacuum(rs, betas, N_cutoff)
# Fw = big_F(w, K, N_cutoff)
#
# half_tanh_N = np.array([(np.tanh(r) / 2)**n for n in range(N_cutoff+1)])
#
# print(np.allclose(p0 * half_tanh_N * Fw, pN))


