import os
import pandas as pd
import random
import numpy as np

class MatrixUtils:

    @staticmethod
    def remove_small(a, tol=1e-7):
        """
        :param a: Any complex np array
        :param tol: Relative tolerance compared to the average
        :return: a with small elements removed
        """

        a_flat = a.flatten()
        av = a_flat.sum() / a_flat.shape[0]

        a.real[abs(a.real) / abs(av.real) < tol] = 0.0
        a.imag[abs(a.imag) / abs(av.imag) <tol] = 0.0

        return a

    @staticmethod
    def filldiag(A, gamma):

        if A.shape[0] != gamma.shape[0]:
            raise ValueError('Input matrix and vector should have compatible shapes')

        for i, gamma_i in enumerate(gamma):
            A[i,i] = gamma_i

        return A

    @staticmethod
    def random_adj(M, max_degree):
        """
        Generate adjacency matrix for random unweighted loopless graph. No restrictions on the possible edges a vertex can have.

        :param M: Number of vertices
        :param max_degree: Maximum degree of the graph

        :return: Adj: Adjacency matrix
        """

        adj = np.zeros((M, M), dtype=int)

        unfilled_vertices = list(range((M)))
        random.shuffle(unfilled_vertices)

        for i, start_vertex in enumerate(unfilled_vertices):
            end_vertices = unfilled_vertices.copy()
            end_vertices.remove(start_vertex)
            start_vertex_degree = adj[start_vertex, :].sum()

            if i == 0:
                # Force 0th vertex to attain max degree, so that the max degree is at least attained once
                num_neighbour = max_degree
            else:
                num_neighbour = np.random.randint(0, min(max_degree + 1 - start_vertex_degree, M - i))
            possible_edge_ends = np.random.choice(end_vertices, num_neighbour, replace=False)

            for possible_edge_end in possible_edge_ends:
                adj[start_vertex, possible_edge_end] = 1
                adj[possible_edge_end, start_vertex] = 1

                end_vertex_degree = adj[possible_edge_end, :].sum()
                assert end_vertex_degree <= max_degree
                if end_vertex_degree == max_degree:
                    unfilled_vertices.remove(possible_edge_end)

            if start_vertex_degree + num_neighbour == max_degree:
                unfilled_vertices.remove(start_vertex)

        return adj


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