import numpy as np
from strawberryfields.decompositions import williamson

from src.symplectic import Symplectic


class GraphMatrices:

    @staticmethod
    def Xmat(M, dtype=np.float64):
        I = np.identity(M, dtype=dtype)
        zero_mat = np.zeros_like(I)
        X = np.block([[zero_mat, I], [I, zero_mat]])

        return X

    @staticmethod
    def cov_Q(cov_fock, dtype=np.complex64):
        M = cov_fock.shape[0] // 2

        return cov_fock + np.identity(2 * M, dtype=dtype) / 2

    @staticmethod
    def Amat(cov_fock, dtype=np.complex64):

        check_valid = GaussianMatrices.is_valid_fock_cov(cov_fock)

        M = cov_fock.shape[0] // 2

        cov_Q = GraphMatrices.cov_Q(cov_fock, dtype)
        X = GraphMatrices.Xmat(M)
        I = np.identity(2 * M, dtype=dtype)

        return X @ (I - np.linalg.inv(cov_Q))


class GaussianMatrices:

    @staticmethod
    def vacuum(M, dtype=np.float64):

        return np.identity(2 * M) / 2

    @staticmethod
    def is_valid_xxpp_cov(cov_xxpp, tol=1e-7):

        (n, m) = cov_xxpp.shape

        cov_xxpp = GaussianMatrices.remove_small(cov_xxpp, tol)

        # Check it is square
        if n != m:
            raise ValueError('Input matrix should be square')

        # Check it has even number of rows and columns
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows and columns')

        # Check it is real
        if not np.all(np.isreal(cov_xxpp)):
            raise ValueError('Input matrix should be real')

        # Check it is symmetric
        if not np.allclose(cov_xxpp, cov_xxpp.T):
            raise ValueError('Input matrix should be symmetric')

        # Check it is positive definite
        eigs = np.linalg.eigvalsh(cov_xxpp)
        if np.any(eigs <= 0):
            raise ValueError('Input matrix should be positive definite')

        # Check it satisfies uncertainty principle
        # the williamson function actually checks for square, symmetric, even number and positive definiteness already
        D, _ = williamson(cov_xxpp)
        symp_eigs = np.diagonal(D)
        if np.any(symp_eigs < 0.5):
            raise ValueError('Input matrix should satisfy uncertainty principle')

        return True

    @staticmethod
    def is_valid_fock_cov(cov_fock):

        # Check it is Hermitian
        if not np.allclose(cov_fock, cov_fock.T.conjugate()):
            raise ValueError('Input matrix should be Hermitian')

        # Convert to xxpp quadrature basis to check the rest, since strawberryfields williamson function only works for
        # real symmetric
        cov_xxpp = Symplectic.matrix_fock_to_xxpp(cov_fock)

        return GaussianMatrices.is_valid_xxpp_cov(cov_xxpp)

    @staticmethod
    def is_valid_fock_mu(mu_fock):
        n = mu_fock.shape[0]

        # Check even number of elements
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        # Check mu_i = mu_(i+M)^*
        mu_split = np.asarray(np.split(mu_fock, 2))
        if not np.allclose(mu_split[[0, 1]], mu_split[[1, 0]].conjugate()):
            raise ValueError('Input vector should be of form [v, v^*]')

        return True

    @staticmethod
    def is_valid_xxpp_mu(mu_xxpp):
        n = mu_xxpp.shape[0]

        # Check even number of elements
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        # Convert to fock basis and check the rest
        mu_fock = Symplectic.vector_xxpp_to_fock(mu_xxpp)

        return GaussianMatrices.is_valid_fock_mu(mu_fock)

    @staticmethod
    def remove_small(a, tol=1e-7):

        a.real[abs(a.real) < tol] = 0.0
        a.imag[abs(a.imag) < tol] = 0.0

        return a