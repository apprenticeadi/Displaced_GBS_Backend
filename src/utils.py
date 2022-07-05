

class MatrixUtils:

    @staticmethod
    def remove_small(a, tol=1e-7):

        a.real[abs(a.real) < tol] = 0.0
        a.imag[abs(a.imag) < tol] = 0.0

        return a

    @staticmethod
    def filldiag(A, gamma):

        if A.shape[0] != gamma.shape[0]:
            raise ValueError('Input matrix and vector should have compatible shapes')

        for i, gamma_i in enumerate(gamma):
            A[i,i] = gamma_i

        return A
