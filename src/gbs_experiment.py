import time
from strawberryfields.decompositions import takagi
import numpy as np
import random
import copy
from scipy.special import factorial

from src.gbs_matrix import GBSMatrix, GaussianMatrix
from src.utils import MatrixUtils, DGBSUtils
from src.symplectic import Symplectic, SymplecticXXPP
from src.photon_number_distributions import single_displaced_squeezed_vacuum, total_displaced_squeezed_vacuum

from thewalrus import hafnian

'''
Construct a Displaced GBS experiment. Object allows conversion between graph matrices, gaussian matrices and 
experimental parameters. One should avoid using Fock basis, except for debugging and testing purposes. 
xxpp basis is preferred as it is real and suffers less numerical errors. 
'''

class PureGBS:

    # Everything in xxpp basis unless otherwise specified, to reduce numerical error.

    def __init__(self, M, cov=None, means=None):
        """
        :param M: Number of modes of the experiment.
        """
        self.M = M
        if cov is None:
            self.__cov = 0.5 * np.identity(2 * M)  # Vacuum state
        else:
            cov = np.asarray(cov)
            if GaussianMatrix.is_valid_cov_xxpp(cov):
                purity = np.linalg.det(cov) * 2 ** (2*M)
                if purity == 1:
                    self.__cov = cov
                else:
                    raise ValueError('Input cov matrix is not pure')

        if means is None:
            self.__means = np.zeros(2 * M)  # Means vector
        else:
            means = np.asarray(means)
            if GaussianMatrix.is_valid_d_xxpp(means):
                self.__means = means

    def state_xxpp(self):
        return self.get_means(), self.get_cov()

    def get_means(self):
        return copy.deepcopy(self.__means)

    def get_cov(self):
        return copy.deepcopy(self.__cov)

    def add_squeezing(self, rs):
        """Add single mode squeezing """
        s = np.absolute(rs)
        theta = np.angle(rs)

        S = SymplecticXXPP.single_mode_squeezing(s, theta)
        self.__means = S @ self.__means
        self.__cov = S @ self.__cov @ S.T

    def add_two_mode_squeezing(self, rs):
        """
        Can only accept real squeezing parameters.
        """
        rs = np.atleast_1d(rs)
        if any(rs.imag != 0):
            raise Warning('Only accept real squeezing parameters. Automatically discard imaginary parts')
        rs = rs.real

        S = SymplecticXXPP.two_mode_squeezing(rs)
        self.__means = S @ self.__means
        self.__cov = S @ self.__cov @ S.T

    def add_interferometer(self, U):
        S = SymplecticXXPP.interferometer(U)
        self.__means = S @ self.__means
        self.__cov = S @ self.__cov @ S.T

    def add_displacement(self, alphas):
        alphas = np.atleast_1d(alphas)
        d = np.sqrt(2) * np.concatenate([alphas.real, alphas.imag])
        self.__means += d

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

    def lhaf(self, outcome, displacement = True):
        outcome = np.atleast_1d(outcome)

        B = self.calc_B()
        half_gamma = self.calc_half_gamma()

        B_n = MatrixUtils.n_repetition(B, outcome)
        half_gamma_n = MatrixUtils.n_repetition(half_gamma, outcome)

        np.fill_diagonal(B_n, half_gamma_n)
        # haf_B = MatrixUtils.filldiag(B_n, half_gamma_n)

        lhafnian = hafnian(B_n, loop=displacement)

        return lhafnian

    def prob(self, outcome, include_vac_prob=True, displacement = True):
        lhafnian = self.lhaf(outcome, displacement)

        prob = np.absolute(lhafnian) ** 2

        if include_vac_prob:
            vacuum_prob = self.vacuum_prob()
            prob = vacuum_prob * prob

        return prob / np.prod(factorial(outcome))


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
        self.__B = None
        self.__alphas = np.zeros(M)
        self.__rs = np.zeros(M)
        self.__U = np.identity(M)
        self.added_sq = False
        self.added_dis = False
        self.added_intf = False


    def get_cov(self):
        B = self.calc_B()
        Id = np.identity(self.M)
        cov_Q_inv = np.block([
            [Id, -B.conjugate()],
            [-B, Id]
        ])  # But this is Fock basis!
        cov_Q_inv = Symplectic.matrix_fock_to_xxpp(cov_Q_inv)
        cov = np.linalg.inv(cov_Q_inv) - np.identity(2*self.M) / 2
        return cov

    def get_means(self):
        return np.sqrt(2) * np.concatenate([self.__alphas.real, self.__alphas.imag])

    def get_alphas(self):
        return copy.deepcopy(self.__alphas)

    def get_rs(self):
        return copy.deepcopy(self.__rs)

    def get_U(self):
        return copy.deepcopy(self.__U)

    def add_squeezing(self, rs):
        """Add single mode squeezing"""
        if self.added_sq:
            raise Exception('You added squeezing already')
        else:
            self.added_sq = True
        rs = np.atleast_1d(rs)
        if any(rs.imag != 0):
            raise Warning('Only accept real squeezing parameters. Automatically discard imaginary parts')
        rs = rs.real
        self.__rs = copy.deepcopy(rs)

    def add_two_mode_squeezing(self, rs):
        raise Exception('Two mode squeezing in SUD model not supported yet')

    def add_interferometer(self, U):
        if self.added_intf:
            raise Exception('You added interferometer already')
        else:
            self.added_intf = True
        self.__U = copy.deepcopy(U)
        # super().add_interferometer(U)  # We want to avoid calculating the cov and means, otherwise no point in the takagi shortcut

    def add_displacement(self, alphas):
        """Add displacement after interferometer"""
        if self.added_dis:
            raise Exception('You added displacement already')
        else:
            self.added_dis = True
        alphas = np.asarray(alphas)
        self.__alphas = copy.deepcopy(alphas)

    def add_all(self, rs, alphas, U):
        self.add_squeezing(rs)
        self.add_interferometer(U)
        self.add_displacement(alphas)

    def calc_B(self):
        # Cautious: numpy arrays are mutable, so instance attributes will change if tamper with the array outside the class!
        if self.__B is None:
            rs = self.get_rs()
            U = self.get_U()
            tanhr = np.tanh(rs)
            B = U @ np.diag(tanhr) @ U.T
            self.__B = copy.deepcopy(B)
        else:
            B = copy.deepcopy(self.__B)

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
        alphas = self.get_alphas()
        half_gamma = alphas.conjugate() - B @ alphas

        return half_gamma

    def generate_unweighted_adj(self):
        """Generates adjacency matrix for unweighted loopless graph on which we calculate the matching polynomial"""

        B = self.calc_B()
        B = MatrixUtils.filldiag(B, np.zeros(self.M))
        adj = (B != 0)

        return adj.astype(int)

    def generate_weighted_adj(self):
        """This is tilde B matrix"""

        B = self.calc_B()
        B = MatrixUtils.filldiag(B, np.zeros(self.M))
        half_gamma = self.calc_half_gamma()
        if np.any(half_gamma == 0):
            raise Exception('Cannot transform into matching polynomial graph as some modes are undisplaced. ')

        x = B / np.outer(half_gamma, half_gamma)
        return x

    def vacuum_prob(self):
        B = self.calc_B()
        alphas = self.get_alphas()
        rs = self.get_rs()

        return np.exp((alphas @ B @ alphas).real - sum(np.absolute(alphas)**2)) / np.prod(np.absolute(np.cosh(rs)))

    # def prob(self, outcome):
    #
    #     B = self.calc_B()
    #     half_gamma = self.calc_half_gamma()
    #     B_n = MatrixUtils.n_repetition(B, outcome)
    #     half_gamma_n = MatrixUtils.n_repetition(half_gamma, outcome)
    #     haf_B = MatrixUtils.filldiag(B_n, half_gamma_n)
    #
    #     vacuum_prob = self.vacuum_prob()
    #
    #     prob = vacuum_prob * np.absolute(hafnian(haf_B, loop=True)) ** 2
    #     return prob / np.prod(factorial(outcome))


class sduGBS(sudGBS):

    def __init__(self, M):

        super().__init__(M)
        self.__betas = np.zeros(M)

    # Override
    def add_displacement(self, betas):
        """Add displacement before interferometer"""

        if self.added_dis:
            raise Exception("You added displacement already")

        betas = np.asarray(betas)
        self.__betas = copy.deepcopy(betas)

        if self.added_intf:
            U = self.get_U()
            alphas = U.conjugate() @ betas
            super().add_displacement(alphas)
        else:
            super().add_displacement(betas)

    def get_betas(self):
        return copy.deepcopy(self.__betas)

    # Override
    def add_interferometer(self, U):
        if self.added_intf:
            raise Exception('You added interferometer already')
        else:
            super().add_interferometer(U)

        if self.added_dis:
            self.added_dis = False
            alphas = self.get_alphas()
            alphas = U.conjugate() @ alphas
            super().add_displacement(alphas)

    # Override
    def vacuum_prob(self):
        betas = self.get_betas()
        rs = self.get_rs()

        exponent = 2 * np.sum(np.abs(betas)**2) - np.sum( np.tanh(rs) * (betas**2 + np.conj(betas)**2) )

        return np.exp(-0.5 * exponent) / np.prod(np.absolute(np.cosh(rs)))

    def photon_number_distribution(self, mode, cutoff):
        """
        Find the photon number distribution of a single mode up to cutoff
        """
        betas = self.get_betas()
        beta = betas[mode]
        rs = self.get_rs()
        r = rs[mode]

        if r == 0:
            raise Exception('Current functions do not support photon number distribution of unsqueezed states')

        return single_displaced_squeezed_vacuum(r, beta, cutoff)

    def create_d_gbs(self, K, N, w, U):
        """
        Creates a DisplacedGBS experiment according to our model of K identical displaced squeezed input, with
        displacement/squeezing ratio w
        """

        N_mean = N / K

        r, beta = DGBSUtils.solve_w(w, N_mean)
        w_calc = beta * (1 - np.tanh(r)) / np.sqrt(np.tanh(r))
        if not np.isclose(w_calc, w):
            raise Warning(f'solve_w FAILED: target w = {w}, calc w = {w_calc}, err={w_calc-w}, N={N}')

        rs = np.zeros(self.M)
        rs[:K] = r

        betas = np.zeros(self.M)
        betas[:K] = beta

        self.add_squeezing(rs)
        self.add_displacement(betas)
        self.add_interferometer(U)

        return r, beta

