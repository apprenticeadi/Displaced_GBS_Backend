from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy

import interferometer as itf

from src.gbs_matrices import GraphMatrices, GaussianMatrices
from src.utils import MatrixUtils

# Construct experiment

class PureGBS:

    def __init__(self, M):
        """
        :param M: Number of modes of the experiment.
        """
        self.M = M
        self._B = None
        self.alphas = np.zeros(M)
        self.rs = np.zeros(M)
        self.U = np.identity(M)

    def add_interferometer(self, U):

        if not GaussianMatrices.is_valid_U(U):
            raise ValueError('Input matrix is not valid unitary interferometer')

        n = U.shape[0]
        if n != self.M:
            raise ValueError('Input matrix should have dimensions (M*M)')

        self.U = U

    def add_squeezing(self, rs):
        rs = np.asarray(rs)
        self.rs = rs

    def add_coherent(self, alphas):
        alphas = np.asarray(alphas)
        self.alphas = alphas

    def calc_B(self):

        # Cautious: numpy arrays are mutable, so instance attributes will change if tamper with the array outside the class!

        if self._B is None:
            D = np.identity(self.M)
            v = np.tanh(self.rs)
            D = MatrixUtils.filldiag(D, v)
            B = self.U @ D @ self.U.T

            self._B = copy.deepcopy(B)
        else:
            B = copy.deepcopy(self._B)

        return B

    def calc_A(self):
        B = self.calc_B()
        return GraphMatrices.pure_A_from_B(B)

    def calc_cov_fock(self):
        A = self.calc_A()

        return GaussianMatrices.cov_fock(A)

    def calc_d_fock(self):
        """
        :return: displacement vector in Fock basis
        """

        return np.concatenate([self.alphas, self.alphas.conjugate()])

    def calc_Gamma(self):

        cov_fock = self.calc_cov_fock()

        return GraphMatrices.Gamma(cov_fock, self.calc_d_fock())

    def calc_half_gamma(self):
        Gamma = self.calc_Gamma()

        return Gamma[:self.M]

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
        half_gamma = self.calc_half_gamma()
        if np.any(half_gamma == 0):
            raise Exception('Cannot transform into matching polynomial graph as some modes are undisplaced. ')

        x = B / np.outer(half_gamma, half_gamma)
        return x

