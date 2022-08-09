from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy

import interferometer as itf

from src.gbs_matrix import GBSMatrix, GaussianMatrix
from src.utils import MatrixUtils
from src.symplectic import SymplecticFock


# Construct experiment

class PureGBS:

    def __init__(self, M, dtype=np.complex64):
        """
        :param M: Number of modes of the experiment.
        """
        self.M = M

        self._cov_fock = 0.5 * np.identity(2 * M, dtype=dtype)  # Vacuum state
        self._means_fock = np.zeros(2 * M, dtype=dtype)  # Means vector

    def add_interferometer(self, U):

        S = SymplecticFock.interferometer(U)
        self._means_fock = S @ self._means_fock
        self._cov_fock = S @ self._cov_fock @ S.T.conjugate()

    def add_squeezing(self, rs):
        s = np.absolute(rs)
        theta = np.angle(rs)

        S = SymplecticFock.single_mode_squeezing(s, theta)
        self._means_fock = S @ self._means_fock
        self._cov_fock = S @ self._cov_fock @ S.T.conjugate()

    def add_two_mode_squeezing(self, r, mode_pair):
        s = np.absolute(r)
        theta = np.angle(r)

        S = SymplecticFock.two_mode_squeezing(s, self.M, mode_pair, theta)
        self._means_fock = S @ self._means_fock
        self._cov_fock = S @ self._cov_fock @ S.T.conjugate()

    def add_displacement(self, alphas):
        d = np.concatenate([alphas, alphas.conjugate()])
        self._means_fock += d

    def state_fock(self):
        return copy.deepcopy(self._means_fock), copy.deepcopy(self._cov_fock)

    def calc_A(self):
        _, cov_fock = self.state_fock()

        A = GBSMatrix.Amat(cov_fock)

        return A

    def calc_B(self):
        A = self.calc_A()
        M = self.M

        return A[:M, :M]


# TODO: inherit TakagiGBS from PureGBS
class TakagiGBS(PureGBS):

    def __init__(self, M, dtype=np.complex64):
        """
        :param M: Number of modes of the experiment.
        """
        super().__init__(M, dtype)
        self.M = M
        self._B = None
        self.alphas = np.zeros(M, dtype=dtype)
        self.rs = np.zeros(M, dtype=dtype)
        self.U = np.identity(M, dtype=dtype)

    def add_interferometer(self, U):

        self.U = U
        super().add_interferometer(U)

    def add_squeezing(self, rs):
        rs = np.asarray(rs)
        self.rs = rs
        super().add_squeezing(rs)

    def add_displacement(self, alphas):
        alphas = np.asarray(alphas)
        self.alphas = alphas
        super().add_displacement(alphas)

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
        half_gamma = self.alphas.conjugate() - self.alphas @ B

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
