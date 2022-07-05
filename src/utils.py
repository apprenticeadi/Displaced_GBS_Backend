import os
import pandas as pd

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


class DFUtils:

    @staticmethod
    def create_filename(filename: str):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename

    @staticmethod
    def read_filename_head(directory, filename_head, idx=0, dt=None):
        files = os.listdir(directory)

        filtered_files = [file_ for file_ in files if file_.startswith(filename_head)]
        file_to_read = os.path.join(directory, filtered_files[idx])

        df = pd.read_csv(
            file_to_read,
            dtype=dt,
        )

        return df, file_to_read