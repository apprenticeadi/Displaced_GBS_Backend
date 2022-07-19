import numpy as np
from strawberryfields.decompositions import williamson

from src.symplectic import Symplectic
from src.utils import MatrixUtils

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

    @staticmethod
    def pure_A_from_B(Bmat, dtype=np.complex64):
        Omat = np.zeros_like(Bmat)
        Amat = np.block([
            [Bmat, Omat],
            [Omat, Bmat.conjugate()]
        ])

        return Amat

    @staticmethod
    def Gamma(cov_fock, d_fock, dtype=np.complex64):
        """
        :param cov_fock: Fock state covariance matrix
        :param d_fock: Fock state displacement vector (alpha_1, ..., alpha_M, alpha_1^*, ..., alpha_M^*)
        :return: The 2M Gamma vector that fills A diagonal when calculating loop Hafnian.
        """

        cov_Q = GraphMatrices.cov_Q(cov_fock)
        return d_fock.conjugate() @ np.linalg.inv(cov_Q)


    @staticmethod
    def is_valid_Amat(A):
        (n,m) = A.shape

        if n != m:
            raise ValueError('Input matrix should be square')
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows/columns')

        M = n // 2
        B1 = A[:M, :M]
        C1 = A[:M, M:]
        C2 = A[M:, :M]
        B2 = A[M:, M:]

        if not np.allclose(A, A.T):  # This will require both C1=C2.T and B1=B2.T
            raise ValueError('Input matrix should be symmetric')

        if not np.allclose(B1, B2.conjugate()):
            raise ValueError('Input matrix should have block diagonal B and B^*')
        if not np.allclose(C1, C1.T.conjugate()):
            raise ValueError('Input matrix should have off diagonal C and C.T, such that C is Hermitian')

        eigs_A = np.linalg.eigvals(A)  # A is not symmetric but not necessarily Hermitian.
        if np.any(abs(eigs_A) >= 1):
            raise ValueError('Input matrix should have eigenvalues with absolute value smaller than 1')

        if not np.allclose(B1@C1, C1@B1):
            raise ValueError('Input matrix submatrix should satisfy B and C commute')

        if np.any(C1 < 0):
            raise ValueError('Input matrix off diagonal block submatrix should satisfy C>=0')

        return True

    @staticmethod
    def is_valid_Gamma(Gamma):
        m = Gamma.shape[0]

        if m % 2 != 0 :
            raise ValueError('Input vector should have even length')

        M = m // 2
        halfgamma_1 = Gamma[:M]
        halfgamma_2 = Gamma[M:]

        if not np.allclose(halfgamma_1, halfgamma_2.conjugate()):
            raise ValueError('Input vector should have form (v, v.conjugate)')

        return True

class GaussianMatrices:

    @staticmethod
    def vacuum(M, dtype=np.float64):

        return np.identity(2 * M) / 2

    @staticmethod
    def cov_fock_from_A(A, dtype=np.complex64):

        if not GraphMatrices.is_valid_Amat(A):
            raise Exception('Input matrix is not valid A matrix')

        M = A.shape[0] // 2
        X = GraphMatrices.Xmat(M, dtype=dtype)

        Sigma_Q_inv = np.identity(2*M, dtype=dtype) - X @ A
        Sigma_Q = np.linalg.inv(Sigma_Q_inv)

        cov_fock = Sigma_Q - np.identity(2*M) / 2

        return cov_fock

    @staticmethod
    def sigma_Q_from_A(A, dtype=np.complex64):
        if not GraphMatrices.is_valid_Amat(A):
            raise Exception('Input matrix is not valid A matrix')

        M = A.shape[0] // 2
        X = GraphMatrices.Xmat(M, dtype=dtype)

        Sigma_Q_inv = np.identity(2*M, dtype=dtype) - X @ A
        Sigma_Q = np.linalg.inv(Sigma_Q_inv)

        return Sigma_Q

    @staticmethod
    def mu_fock_from_A(A, gamma, dtype=np.complex64):

        if A.shape[0] != gamma.shape[0]:
            raise ValueError('Input matrix and vector should have compatible shape')

        sigma_Q = GaussianMatrices.sigma_Q_from_A(A, dtype=dtype)

        d_conj = gamma @ sigma_Q

        return d_conj.conjugate()

    @staticmethod
    def is_valid_xxpp_cov(cov_xxpp, tol=1e-7):

        (n, m) = cov_xxpp.shape

        cov_xxpp = MatrixUtils.remove_small(cov_xxpp, tol)

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
        eigs = np.linalg.eigvalsh(cov_xxpp)  # cov_xxpp should be Hermitian
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
    def is_valid_U(U, tol=1e-7, dtype=np.complex64):

        (n, m) = U.shape

        U = MatrixUtils.remove_small(U, tol=tol)

        if n != m:
            raise ValueError('Input matrix should be square')

        Id = np.identity(n, dtype=dtype)

        if not np.allclose(U @ U.T.conjugate(), Id):
            raise ValueError('Input matrix should be unitary')

        return True