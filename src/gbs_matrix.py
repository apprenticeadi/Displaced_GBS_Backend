import numpy as np
from strawberryfields.decompositions import williamson

from src.symplectic import Symplectic
from src.utils import MatrixUtils

'''
Methods to convert between the graph formalism and Gaussian state formalism of Displaced GBS. 
One should almost always avoid using the Fock basis matrices, but they are useful for debugging and testing.
'''

class GBSMatrix:
    """
    GBS Matrices refer to the matrices and vectors that appear in the loop Hafnian calculation. Specifically they are
    the A marix, the B matrix and the Gamma vector.
    This class gives functions for generating GBS matrices from Fock basis Gaussian matrices.
    """

    @staticmethod
    def Xmat(M, dtype=np.float64):
        """ Returns M*M matrix [[0, Id],[Id, 0]] """
        I = np.identity(M, dtype=dtype)
        zero_mat = np.zeros_like(I)
        X = np.block([[zero_mat, I], [I, zero_mat]])

        return X

    @staticmethod
    def Amat(cov_xxpp):

        if not GaussianMatrix.is_valid_cov_xxpp(cov_xxpp):
            raise ValueError('Input matrix is not valid xxpp-base covariance matrix')

        cov_fock = Symplectic.matrix_xxpp_to_fock(cov_xxpp)

        M = cov_fock.shape[0] // 2

        cov_Q = cov_fock + np.identity(2 * M) / 2
        X = GBSMatrix.Xmat(M)
        I = np.identity(2 * M)

        return X @ (I - np.linalg.inv(cov_Q))

    @staticmethod
    def pure_A_from_B(B):
        B = np.asarray(B)

        Omat = np.zeros_like(B)
        Amat = np.block([
            [B, Omat],
            [Omat, B.conjugate()]
        ])

        return Amat

    @staticmethod
    def Gamma(cov_xxpp, means_xxpp):
        """
        :param cov_xxpp: xxpp-basis covariance matrix
        :param means_xxpp: xxpp-basis displacement vector sqrt(2) * (Re(alpha_1), ..., Re(alpha_M), Im(alpha_1), ..., Im(alpha_M))
        :return: The 2M Gamma vector that fills A diagonal when calculating loop Hafnian.
        """
        cov_xxpp = np.asarray(cov_xxpp)
        if not GaussianMatrix.is_valid_cov_xxpp(cov_xxpp):
            raise ValueError('Input matrix is not valid xxpp-basis covariance matrix')
        means_xxpp = np.asarray(means_xxpp)
        if not GaussianMatrix.is_valid_d_xxpp(means_xxpp):
            raise ValueError('Input vector is not valid xxpp-basis means vector')

        M = cov_xxpp.shape[0] // 2

        if means_xxpp.shape[0] != 2 * M:
            raise ValueError('Input matrix and vector should have compatible shape')

        cov_fock = Symplectic.matrix_xxpp_to_fock(cov_xxpp)
        means_fock = Symplectic.vector_xxpp_to_fock(means_xxpp)

        cov_Q = cov_fock + np.identity(2 * M) / 2
        return means_fock.conjugate() @ np.linalg.inv(cov_Q)

    @staticmethod
    def is_valid_Amat(A):
        A = np.asarray(A)
        (n, m) = A.shape

        if n != m:
            raise ValueError('Input matrix should be square')
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows/columns')

        M = n // 2
        B1 = A[:M, :M]
        C1 = A[:M, M:]
        # C2 = A[M:, :M]
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

        if not np.allclose(B1 @ C1, C1 @ B1):
            raise ValueError('Input matrix submatrix should satisfy B and C commute')

        if np.any(C1 < 0):
            raise ValueError('Input matrix off diagonal block submatrix should satisfy C>=0')

        return True

    @staticmethod
    def is_pure_Amat(Amat):
        Amat = np.asarray(Amat)
        M = Amat.shape[0] // 2

        if not GBSMatrix.is_valid_Amat(Amat):  # If this step doesn't return error, then off diagonal C1=C2.T
            raise ValueError('Input matrix is not valid A matrix')

        C1 = Amat[:M, M:]
        Omat = np.zeros_like(C1)

        if np.allclose(C1, Omat):
            return True
        else:
            return False

    @staticmethod
    def is_valid_Gamma(Gamma):
        Gamma = np.asarray(Gamma)
        m = Gamma.shape[0]

        if m % 2 != 0:
            raise ValueError('Input vector should have even length')

        M = m // 2
        halfgamma_1 = Gamma[:M]
        halfgamma_2 = Gamma[M:]

        if not np.allclose(halfgamma_1, halfgamma_2.conjugate()):
            raise ValueError('Input vector should have form (v, v.conjugate)')

        return True


class GaussianMatrix:
    """Class of functions for generating covariance matrix and means vector from GBS matrices"""

    @staticmethod
    def vacuum(M, dtype=np.float64):

        return np.identity(2 * M, dtype=dtype) / 2

    @staticmethod
    def cov_fock(A):
        A = np.asarray(A)

        if not GBSMatrix.is_valid_Amat(A):
            raise Exception('Input matrix is not valid A matrix')

        M = A.shape[0] // 2
        X = GBSMatrix.Xmat(M)

        cov_Q_inv = np.identity(2 * M) - X @ A
        cov_Q = np.linalg.inv(cov_Q_inv)

        cov_fock = cov_Q - np.identity(2 * M) / 2

        return cov_fock

    @staticmethod
    def cov_xxpp(A):
        cov_fock = GaussianMatrix.cov_fock(A)

        return Symplectic.matrix_fock_to_xxpp(cov_fock).real

    @staticmethod
    def cov_Q(A):
        A = np.asarray(A)

        if not GBSMatrix.is_valid_Amat(A):
            raise Exception('Input matrix is not valid A matrix')

        M = A.shape[0] // 2
        X = GBSMatrix.Xmat(M)

        cov_Q_inv = np.identity(2 * M) - X @ A
        cov_Q = np.linalg.inv(cov_Q_inv)

        return cov_Q

    @staticmethod
    def d_fock(A, Gamma):
        A = np.asarray(A)

        if A.shape[0] != Gamma.shape[0]:
            raise ValueError('Input matrix and vector should have compatible shape')

        cov_Q = GaussianMatrix.cov_Q(A)

        return cov_Q @ Gamma.conjugate()

    @staticmethod
    def d_xxpp(A, Gamma):
        d_fock = GaussianMatrix.d_fock(A, Gamma)

        return Symplectic.vector_fock_to_xxpp(d_fock)

    @staticmethod
    def is_valid_cov_xxpp(cov_xxpp, tol=1e-7):
        cov_xxpp = np.asarray(cov_xxpp)

        (n, m) = cov_xxpp.shape

        # Check it is square
        if n != m:
            raise ValueError('Input matrix should be square')

        # Check it has even number of rows and columns
        if n % 2 != 0:
            raise ValueError('Input matrix should have even number of rows and columns')

        # Check it is real
        if not np.all(np.isreal(MatrixUtils.remove_small(cov_xxpp, tol))):
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
        if np.any(symp_eigs + tol < 0.5):  # TODO: is this valid?
            raise ValueError('Input matrix should satisfy uncertainty principle')

        return True

    #TODO: This method never works, because converting to xxpp still leaves some imaginary parts that is hard to remove.
    @staticmethod
    def is_valid_cov_fock(cov_fock, tol=1e-7):
        raise Warning('This method never works' )

        cov_fock = np.asarray(cov_fock)

        # Check it is Hermitian
        if not np.allclose(cov_fock, cov_fock.T.conjugate()):
            raise ValueError('Input matrix should be Hermitian')

        # Convert to xxpp quadrature basis to check the rest, since strawberryfields williamson function only works for
        # real symmetric
        #TODO: this is bad practice.
        cov_xxpp = Symplectic.matrix_fock_to_xxpp(cov_fock).real


        return GaussianMatrix.is_valid_cov_xxpp(cov_xxpp, tol=tol)

    @staticmethod
    def is_valid_d_fock(d_fock):
        d_fock = np.asarray(d_fock)
        n = d_fock.shape[0]

        # Check even number of elements
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        # Check mu_i = mu_(i+M)^*
        d_split = np.asarray(np.split(d_fock, 2))
        if not np.allclose(d_split[[0, 1]], d_split[[1, 0]].conjugate()):
            raise ValueError('Input vector should be of form [v, v^*]')

        return True

    @staticmethod
    def is_valid_d_xxpp(d_xxpp):
        d_xxpp = np.asarray(d_xxpp)
        n = d_xxpp.shape[0]

        # Check even number of elements
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        # No need to convert to Fock basis and check again, because multiplying with Lmat gurantees that d_fock
        # has the form [v, v^*]
        # d_fock = Symplectic.vector_xxpp_to_fock(d_xxpp)

        return True

    @staticmethod
    def is_valid_U(U, tol=1e-7):
        U = np.asarray(U)

        (n, m) = U.shape

        # Not sure if this line should be here
        # U = MatrixUtils.remove_small(U, tol=tol)

        if n != m:
            raise ValueError('Input matrix should be square')

        Id = np.identity(n)

        if not np.allclose(U @ U.T.conjugate(), Id):
            raise ValueError('Input matrix should be unitary')

        return True
