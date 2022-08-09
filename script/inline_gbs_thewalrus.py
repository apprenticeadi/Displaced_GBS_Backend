from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
import numpy as np

from thewalrus.symplectic import vacuum_state, squeezing, two_mode_squeezing, interferometer, expand
from thewalrus.quantum import Amat


def takagi_circuit(M, r, U, alpha):
    mu, cov = tuple(vacuum_state(M, hbar=1.0))

    S1 = squeezing(np.absolute(r), np.angle(r))
    mu = S1 @ mu
    cov = S1 @ cov @ S1.T

    Su = interferometer(U)
    mu = Su @ mu
    cov = Su @ cov @ Su.T

    mu += np.concatenate([alpha.real, alpha.imag])

    return mu, cov


def inline_circuit(M, B, half_gamma, v=None):
    mu, cov = tuple(vacuum_state(M, hbar=1.0))

    B = np.asarray(B)
    half_gamma = np.asarray(half_gamma)
    if v is None:
        v = B.diagonal()

    # Displacement
    mu += np.sqrt(2) * np.concatenate([half_gamma.real, half_gamma.imag])

    # Single mode squeezing
    S1 = squeezing(np.absolute(v), np.angle(v))
    mu = S1 @ mu
    cov = S1 @ cov @ S1.T

    for j in range(M):
        for k in range(j + 1, M):
            Bjk = B[j, k]
            S2 = two_mode_squeezing(np.absolute(Bjk), np.angle(Bjk))

            S_exp = expand(S2, [j, k], M)

            mu = S_exp @ mu
            cov = S_exp @ cov @ S_exp.T

    return mu, cov


M = 4
U = unitary_group.rvs(M)
r = np.asarray([0.5] * M)
alpha = np.asarray([2] * M)


# minus sign on r because thewalrus uses a different single mode
# squeezing operator as in the notes
mu1, cov1 = takagi_circuit(M, -r, U, alpha)

A = Amat(cov1, hbar=1.0)
B = A[:M, :M]
v = B.diagonal()
half_gamma = alpha.conjugate() - alpha @ B

mu2, cov2 = inline_circuit(M, B, half_gamma, -v)
