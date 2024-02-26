import numpy as np
from scipy.linalg import block_diag
from strawberryfields.decompositions import williamson

class Symplectic:

    @staticmethod
    def Lmat(M, dtype=np.complex64):
        I = np.identity(M, dtype=dtype)
        L = np.block([[I, 1j * I], [I, -1j * I]]) / np.sqrt(2)

        return L

    @staticmethod
    def Omegamat(M, dtype=np.float64):
        I = np.identity(M, dtype=dtype)
        O = np.zeros_like(I)

        Omega = np.block([[O, I], [-I, O]])

        return Omega

    @staticmethod
    def matrix_fock_to_xxpp(matrix_fock):
        matrix_fock = np.asarray(matrix_fock)
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
        vector_fock=np.asarray(vector_fock)
        n = vector_fock.shape[0]
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')
        M = n // 2
        Lmat = Symplectic.Lmat(M)
        vector_xxpp = Lmat.T.conjugate() @ vector_fock

        return vector_xxpp


    @staticmethod
    def matrix_xxpp_to_fock(matrix_xxpp):
        matrix_xxpp = np.asarray(matrix_xxpp)
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
        vector_xxpp = np.asarray(vector_xxpp)
        n = vector_xxpp.shape[0]
        if n % 2 != 0:
            raise ValueError('Input vector should have even number of elements')

        M = n // 2
        Lmat = Symplectic.Lmat(M)
        vector_fock = Lmat @ vector_xxpp

        return vector_fock

class SymplecticFock(Symplectic):
    # This class borrows from thewalrus.symplectic, but in Fock basis instead.

    @staticmethod
    def single_mode_squeezing(s, theta=None, dtype=np.complex64):
        r"""
        The operator is
        $$ \bigotimes_{i=1}^M \exp \left[ \frac{1}{2}( r\hat{a}_i^{\dagger2} - r^*\hat{a}_i^2 ) \right]$$
        where
        $$r = s e^{i\theta}$$

        :param s: Squeezing amplitude, $s=|r|$
        :param theta: Squeezing phase angle $\theta$
        :param dtype:
        :return: Fock basis symplectic matrix for single mode squeezing
        """

        s = np.atleast_1d(s)  # converts inputs into arrays of at least 1 dimension

        if theta is None:
            theta = np.zeros_like(s)

        M = len(s)
        S = np.identity(2*M, dtype = dtype)

        for i, (s_i, theta_i) in enumerate(zip(s, theta)):
            S[i,i] = np.cosh(s_i)
            S[i, i+M] = np.exp(1j * theta_i) * np.sinh(s_i)
            S[i+M, i] = np.exp(-1j * theta_i) * np.sinh(s_i)
            S[i+M, i+M] = np.cosh(s_i)

        return S

    @staticmethod
    def two_mode_squeezing(s, M, mode_pair, theta=None, dtype=np.complex64):
        r"""
        The operator is
        $$ \exp \left[ r\hat{a}_i^{\dagger}\hat{a}_j^\dagger -  r^*\hat{a}_i\hat{a}_j \right]$$
        where
        $$r = s e^\theta$$

        :param s: Squeezing amount $|r|$
        :param M: Total number of modes
        :param mode_pair: Tuple of mode indices (i, j)
        :param theta: Squeezing phase angle $\theta$
        :param dtype:
        :return: 2M*2M Fock basis symplectic matrix. i-th, j-th, i+M-th and j+M-th rows and columns performs the squeezing, while
        the rest is identity.
        """

        i, j = mode_pair
        S = np.identity(2*M, dtype=dtype)

        S[i,i] = np.cosh(s)
        S[j,j] = np.cosh(s)
        S[i+M, i+M] = np.cosh(s)
        S[j+M, j+M] = np.cosh(s)

        S[i, j+M] = np.exp(1j * theta) * np.sinh(s)
        S[j, i+M] = np.exp(1j * theta) * np.sinh(s)
        S[i+M, j] = np.exp(-1j * theta) * np.sinh(s)
        S[j+M, i] = np.exp(-1j * theta) * np.sinh(s)

        return S

    @staticmethod
    def interferometer(U):
        """
        Fock basis symplectic matrix for interferometer
        :param U: unitary matrix
        :return: symplectic transformation matrix (Fock basis)
        """

        M = U.shape[0]

        if not np.allclose(U @ U.T.conjugate(), np.identity(M)):
            raise ValueError('Input matrix should be unitary')

        O = np.zeros_like(U)
        S = np.block([[U.conjugate(), O], [O, U]])

        return S


class SymplecticXXPP(Symplectic):


    @staticmethod
    def single_mode_squeezing(s, theta=None, dtype=np.float64):
        r"""
        The operator is
        $$ \bigotimes_{i=1}^M \exp \left[ \frac{1}{2}( r\hat{a}_i^{\dagger2} - r^*\hat{a}_i^2 ) \right]$$
        where
        $$r = s e^{i\theta}$$

        :param s: Squeezing amplitude, $s=|r|$
        :param theta: Squeezing phase angle $\theta$
        :param dtype:
        :return: xxpp-basis symplectic matrix for single mode squeezing
        """

        s = np.atleast_1d(s)  # converts inputs into arrays of at least 1 dimension

        if theta is None:
            theta = np.zeros_like(s)

        M = len(s)
        S = np.identity(2 * M, dtype=dtype)

        for i, (s_i, theta_i) in enumerate(zip(s, theta)):
            S[i, i] = np.cosh(s_i) + np.cos(theta_i) * np.sinh(s_i)
            S[i, i + M] = np.sin(theta_i) * np.sinh(s_i)
            S[i + M, i] = np.sin(theta_i) * np.sinh(s_i)
            S[i + M, i + M] = np.cosh(s_i) - np.cos(theta_i) * np.sinh(s_i)

        return S

    @staticmethod
    def two_mode_squeezing(s,  dtype=np.complex64):
        r"""
        The j_th mode operator is
        $$ \exp \left[ s_j (\hat{a}_{jH}^{\dagger}\hat{a}_{jV}^\dagger -  \hat{a}_{jH}\hat{a}_{jV}) \right]$$
        Here I made life easier by demanding squeezing parameter, $s_j$, be real.
        H and V stands for horizontal and vertical polarisation.

        :param s: Squeezing amount
        :param dtype:
        :return: 2M*2M xxpp-basis symplectic matrix. The ordering is $\{x_{1H}, x_{1V}, ..., x_{MH}, x_{MV}, p_{1H},
          p_{1V}, ..., p_{MH}, p_{MV} \} $
        """
        s = np.atleast_1d(s)  # converts inputs into arrays of at least 1 dimension
        # M = len(s)

        subs1 = []
        subs2 = []
        for s_i in s:
            Ch = np.array([[np.cosh(s_i), 0], [0, np.cosh(s_i)]])
            Sh = np.array([[0, np.sinh(s_i)], [np.sinh(s_i), 0]])
            subs1.append(Ch+Sh)
            subs2.append(Ch-Sh)

        S = block_diag(*subs1, *subs2)

        return S


    @staticmethod
    def interferometer(U):
        """
        xxpp basis symplectic matrix for interferometer
        :param U: unitary matrix
        :return: symplectic transformation matrix (Fock basis)
        """

        M = U.shape[0]

        if not np.allclose(U @ U.T.conjugate(), np.identity(M)):
            raise ValueError('Input matrix should be unitary')

        X = U.real
        Y = U.imag
        S = np.block([[X, Y], [-Y, X]])

        return S