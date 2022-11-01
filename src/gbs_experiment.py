from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy

import interferometer as itf

from src.gbs_matrix import GBSMatrix, GaussianMatrix
from src.utils import MatrixUtils
from src.symplectic import Symplectic, SymplecticXXPP


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

    def state_xxpp(self):
        return self.get_means(), self.get_cov()

    def get_means(self):
        return copy.deepcopy(self._means)

    def get_cov(self):
        return copy.deepcopy(self._cov)

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

    def add_interferometer(self, U):
        S = SymplecticXXPP.interferometer(U)
        self._means = S @ self._means
        self._cov = S @ self._cov @ S.T

    def add_displacement(self, alphas):
        alphas = np.atleast_1d(alphas)
        d = np.sqrt(2) * np.concatenate([alphas.real, alphas.imag])
        self._means += d

    def calc_A(self):
        cov = self.get_cov()
        return GBSMatrix.Amat(cov)

    def calc_B(self):
        A = self.calc_A()
        M = self.M

        return A[:M, :M]

    def calc_Gamma(self):
        means, cov = self.state_xxpp()
        return GBSMatrix.Gamma(cov, means)

    def calc_half_gamma(self):
        Gamma = self.calc_Gamma()
        return Gamma[:self.M]

    def vacuum_prob(self):
        means, cov = self.state_xxpp()
        cov_Q = cov + np.identity(2 * self.M) / 2  # this is in xxpp basis
        cov_Q_inv = np.linalg.inv(cov_Q)

        return np.exp(-0.5 * means.T @ cov_Q_inv @ means) / np.sqrt(np.linalg.det(cov_Q))


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

    # def state_xxpp(self):
    #     return self.get_means(), self.get_cov()

    def get_cov(self):
        A = self.calc_A()
        return GaussianMatrix.cov_xxpp(A)

    def get_means(self):
        return np.sqrt(2) * np.concatenate([self.alphas.real, self.alphas.imag])

    def add_squeezing(self, rs):
        """Add single mode squeezing"""
        rs = np.atleast_1d(rs)
        if any(rs.imag != 0):
            raise Warning('Only accept real squeezing parameters. Automatically discard imaginary parts')
        rs = rs.real
        self.rs = rs
        # super().add_squeezing(rs)

    def add_two_mode_squeezing(self, rs):
        raise Exception('Two mode squeezing in SUD model not supported yet')

    def add_interferometer(self, U):

        self.U = U
        # super().add_interferometer(U)  # We want to avoid calculating the cov and means, otherwise no point in the takagi shortcut

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
        O = np.zeros_like(B)
        A = np.block([
            [B, O],
            [O, B.conjugate()]
        ])

        return A

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

    def vacuum_prob(self):
        B = self.calc_B()
        Id = np.identity(self.M)
        cov_Q_inv = np.block([
            [Id, -B.conjugate()],
            [-B, Id]
        ])  # But this is Fock basis!
        cov_Q_inv = Symplectic.matrix_fock_to_xxpp(cov_Q_inv)
        means = self.get_means()

        return np.exp(-0.5 * means.T @ cov_Q_inv @ means) * np.sqrt(np.linalg.det(cov_Q_inv))


class sduGBS(sudGBS):

    # Override
    def add_displacement(self, betas):
        """Add displacement before interferometer"""
        betas = np.asarray(betas)
        self.alphas = self.U.conjugate() @ betas

    # Override
    def add_interferometer(self, U):
        self.U = U
        self.alphas = U.conjugate() @ self.alphas
