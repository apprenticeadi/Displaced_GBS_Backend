import numpy as np
from strawberryfields.decompositions import williamson

class Symplectic:

    @staticmethod
    def Lmat(M, dtype=np.complex64):
        I = np.identity(M, dtype=dtype)
        L = np.block([[I, 1j * I], [I, -1j * I]]) / np.sqrt(2)

        return L

    @staticmethod
    def matrix_fock_to_xxpp(matrix_fock):
        (n,m) = matrix_fock.shape
        if n != m :
            raise ValueError('Input matrix should be square')
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows')

        M = n // 2
        L = Symplectic.Lmat(M)
        matrix_xxpp = L.T.conjugate() @ matrix_fock @ L

        return matrix_xxpp

    @staticmethod
    def vector_fock_to_xxpp(vector_fock):
        n = vector_fock.shape[0]
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')
        M = n // 2
        Lmat = Symplectic.Lmat(M)
        vector_xxpp = Lmat.T.conjugate() @ vector_fock

        return vector_xxpp


    @staticmethod
    def matrix_xxpp_to_fock(matrix_xxpp):
        (n, m) = matrix_xxpp.shape
        if n != m:
            raise ValueError('Input matrix should be square')
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows')

        M = n // 2
        Lmat = Symplectic.Lmat(M)
        cov_fock = Lmat @ matrix_xxpp @ Lmat.T.conjugate()

        return cov_fock

    @staticmethod
    def vector_xxpp_to_fock(vector_xxpp):
        n = vector_xxpp.shape[0]
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        M = n // 2
        Lmat = Symplectic.Lmat(M)
        vector_fock = Lmat @ vector_xxpp

        return vector_fock

class SymplecticFock(Symplectic):
    # Todo: This class borrows from thewalrus.symplectic, but in Fock basis instead. Need to check license.

    @staticmethod
    def single_mode_squeezing(s, theta=None, dtype=np.complex64):
        s = np.atleast_1d(s)  # converts inputs into arrays of at least 1 dimension

        if theta is None:
            theta = np.zeros_like(s)

        M = len(s)
        S = np.identity(2*M, dtype = dtype)

        for i, (s_i, theta_i) in enumerate(zip(s, theta)):
            S[i,i] = np.cosh(s_i)
            S[i, i+M] = -np.exp(1j * theta_i) * np.sinh(s_i)
            S[i+M, i] = -np.exp(-1j * theta_i) * np.sinh(s_i)
            S[i+M, i+M] = np.cosh(s_i)

        return S

    # @staticmethod
    # def two_mode_squeezing(s, theta=None, dtype=np.float64):

class SymplecticXXPP(Symplectic):

    @staticmethod
    def single_mode_squeezing(s, theta=None, dtype=np.float64):
        # Todo: this function the same as thewalrus.symplectic.squeezing

        s = np.atleast_1d(s)  # converts inputs into arrays of at least 1 dimension

        if theta is None:
            theta = np.zeros_like(s)

        M = len(s)
        S = np.identity(2 * M, dtype=dtype)

        for i, (s_i, theta_i) in enumerate(zip(s, theta)):
            S[i, i] = np.cosh(s_i) - np.cos(theta_i) * np.sinh(s_i)
            S[i, i + M] = -np.sin(theta_i) * np.sinh(s_i)
            S[i + M, i] = -np.sin(theta_i) * np.sinh(s_i)
            S[i + M, i + M] = np.cosh(s_i) + np.cos(theta_i) * np.sinh(s_i)

        return S


