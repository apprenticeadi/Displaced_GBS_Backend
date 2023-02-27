import numpy as np
from scipy.special import factorial
from numpy.polynomial.hermite import hermval

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

def total_displaced_squeezed_vacuum(rs, betas, K, cutoff):
    """
    Find the total photon number distribution of K single-mode displaced squeezed vacuua up to total cutoff.
    If rs and betas has length 1, then calculate over K identical single-mode states.
    """
    rs = np.atleast_1d(rs)
    betas = np.atleast_1d(betas)
    if len(rs) == len(betas) and len(rs) == 1:
        # identical single mode states
        r = rs[0]
        beta = betas[0]

        p0 = single_displaced_squeezed_vacuum(r, beta, cutoff)
        p_n = np.zeros(cutoff+1)
        p_n[0] = 1
        for iter in range(K):
            p_n = np.convolve(p_n, p0)[:cutoff+1]

    elif len(rs) == len(betas) and len(rs) == K:
        p_n = np.ones(cutoff+1)
        p_n[0] = 1
        for iter in range(K):
            r = rs[iter]
            beta = betas[iter]

            p = single_displaced_squeezed_vacuum(r, beta, cutoff)
            p_n = np.convolve(p_n, p)[:cutoff+1]

    else:
        raise ValueError('Input array dimensions not match')

    return p_n




