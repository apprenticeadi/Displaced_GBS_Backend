import numpy as np

from thewalrus import hafnian

from src.utils import MatrixUtils

def lhaf_squared(U, w, N, loop=True):

    B = U @ U.T
    half_gamma = w * np.sum(U, axis=1)

    np.fill_diagonal(B, half_gamma)

    lhaf = hafnian(B[:N, :N], loop=loop)

    return np.absolute(lhaf) ** 2