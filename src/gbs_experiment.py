from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy

import interferometer as itf

from src.gbs_matrix import GBSMatrix, GaussianMatrix
from src.utils import MatrixUtils
from src.symplectic import SymplecticXXPP


# Construct experiment

# TODO: One should avoid using Fock basis. xxpp basis is real and suffers less numerical errors. #]
#  Converting between fock and xxpp basis is prone to error.

class PureGBS:

    # Everything in xxpp basis unless otherwise specified, to reduce numerical error.

    def __init__(self, M):
        """
        :param M: Number of modes of the experiment.
        """
        self.M = M

        self._cov = 0.5 * np.identity(2 * M)  # Vacuum state
        self._means = np.zeros(2 * M)  # Means vector

    def add_interferometer(self, U):

        S = SymplecticXXPP.interferometer(U)
        self._means = S @ self._means
        self._cov = S @ self._cov @ S.T

    def add_squeezing(self, rs):
        """Add single mode squeezing """
        s = np.absolute(rs)
        theta = np.angle(rs)

        S = SymplecticXXPP.single_mode_squeezing(s, theta)
        self._means = S @ self._means
        self._cov = S @ self._cov @ S.T

    def add_two_mode_squeezing(self, rs):
        """
        Can only accept real squeezing parameters.
        """
        rs = np.atleast_1d(rs)
        if any(rs.imag != 0):
            raise Warning('Only accept real squeezing parameters. Automatically discard imaginary parts')
        rs = rs.real

        S = SymplecticXXPP.two_mode_squeezing(rs)
        self._means = S @ self._means
        self._cov = S @ self._cov @ S.T

    def add_displacement(self, alphas):
        alphas = np.atleast_1d(alphas)
        d = np.sqrt(2) * np.concatenate([alphas.real, alphas.imag])
        self._means += d

    def state_xxpp(self):
        return copy.deepcopy(self._means), copy.deepcopy(self._cov)

    def calc_A(self):
        _, cov_xxpp = self.state_xxpp()

        A = GBSMatrix.Amat(cov_xxpp)

        return A

    def calc_B(self):
        A = self.calc_A()
        M = self.M

        return A[:M, :M]

    def calc_Gamma(self):
        means_xxpp, cov_xxpp = self.state_xxpp()
        return GBSMatrix.Gamma(cov_xxpp, means_xxpp)

    def calc_half_gamma(self):
        Gamma = self.calc_Gamma()
        return Gamma[:self.M]

#Deprecation
def TakagiGBS(*args, **kwargs):
    from warnings import warn
    warn("TakagiGBS deprecated. Get with the program!")
    return sudGBS(*args, **kwargs)


class sudGBS(PureGBS):

    def __init__(self, M):
        """
        :param M: Number of modes of the experiment.
        """
        super().__init__(M)
        self.M = M
        self._B = None
        self.alphas = np.zeros(M)
        self.rs = np.zeros(M)
        self.U = np.identity(M)

    def _calc_cov(self):
        A = self.calc_A()
        self._cov = GaussianMatrix.cov_xxpp(A)

    def _calc_means(self):
        self._means = np.sqrt(2) * np.concatenate([self.alphas.real, self.alphas.imag])

    def state_xxpp(self):
        self._calc_means()
        self._calc_cov()
        return copy.deepcopy(self._means), copy.deepcopy(self._cov)

    def add_interferometer(self, U):

        self.U = U
        # super().add_interferometer(U)  # We want to avoid calculating the cov and means, otherwise no point in the takagi shortcut

    def add_squeezing(self, rs):
        """Add single mode squeezing"""
        rs = np.atleast_1d(rs)
        if any(rs.imag != 0):
            raise Warning('Only accept real squeezing parameters. Automatically discard imaginary parts')
        rs = rs.real
        self.rs = rs
        # super().add_squeezing(rs)

    def add_displacement(self, alphas):
        """Add displacement after interferometer"""
        alphas = np.asarray(alphas)
        self.alphas = alphas
        # super().add_displacement(alphas)

    def calc_B(self):

        # Cautious: numpy arrays are mutable, so instance attributes will change if tamper with the array outside the class!

        if self._B is None:
            Id = np.identity(self.M)
            v = np.tanh(self.rs)
            D = MatrixUtils.filldiag(Id, v)
            B = self.U @ D @ self.U.T

            self._B = copy.deepcopy(B)
        else:
            B = copy.deepcopy(self._B)

        return B

    def calc_A(self):
        B = self.calc_B()
        return GBSMatrix.pure_A_from_B(B)

    def calc_Gamma(self):
        half_gamma = self.calc_half_gamma()

        return np.concatenate([half_gamma, half_gamma.conjugate()])

    def calc_half_gamma(self):
        B = self.calc_B()
        half_gamma = self.alphas.conjugate() - B.conjugate() @ self.alphas

        return half_gamma

    def generate_unweighted_adj(self):
        """Generates adjacency matrix for unweighted loopless graph on which we calculate the matching polynomial"""

        B = self.calc_B()
        B = MatrixUtils.filldiag(B, np.zeros(self.M))
        adj = (B != 0)

        return adj.astype(int)

    def generate_weighted_adj(self):
        """Generates adjacency matrix for weighted loopless graph on which we calculate the matching polynomial,
        in other words this is the edge activities matrix x"""

        B = self.calc_B()
        B = MatrixUtils.filldiag(B, np.zeros(self.M))
        half_gamma = self.calc_half_gamma()
        if np.any(half_gamma == 0):
            raise Exception('Cannot transform into matching polynomial graph as some modes are undisplaced. ')

        x = B / np.outer(half_gamma, half_gamma)
        return x


class sduGBS(sudGBS):

    # Override
    def add_displacement(self, betas):
        """Add displacement before interferometer"""
        betas = np.asarray(betas)
        self.alphas = betas

    # Override
    def add_interferometer(self, U):
        self.U = U
        self.alphas = U.conjugate() @ self.alphas
