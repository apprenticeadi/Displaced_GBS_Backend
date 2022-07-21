from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy

import interferometer as itf

from src.gbs_matrices import GraphMatrices, GaussianMatrices
from src.utils import MatrixUtils

class PureGBS:

    def __init__(self, M):
        """
        :param M: Number of modes of the experiment.
        """
        self.M = M
        self._B = None

    @staticmethod
    def random_interferometer(M, depth):
        I = itf.Interferometer()

        for k in range(depth):
            p = M // 2
            q = M % 2
            if k % 2 != 0 and q == 0:
                shift = 1
            else:
                shift = 0

            for i in range(p - shift):
                j = 2 * i + 1 + k % 2  # Clements interferometer mode index starts from 1
                phase = 0.5 * random.random() * np.pi
                angle = 0.5 * random.random() * np.pi

                bs = itf.Beamsplitter(j, j + 1, angle, phase)

                I.add_BS(bs)

        return I

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

        return GaussianMatrices.cov_fock_from_A(A)

    def calc_d_fock(self):
        """
        :return: displacement vector in Fock basis
        """

        return np.concatenate([self.alphas, self.alphas.conjugate()])

    def calc_Gamma(self):

        cov_fock = self.calc_cov_fock()

        return GraphMatrices.Gamma(cov_fock, self.calc_d_fock())





