import networkx as nx
import numpy as np
import copy

# This is intended to be a wrapper function around networkx, because I can't find a function that goes from nx.Graph to
# matrix.

class AdjacencyMatrix:

    def __init__(self, adj):
        adj = np.asarray(adj)
        m, n = adj.shape
        if m != n:
            raise ValueError('Input adjacency matrix must be square matrix')

        self.__adj = np.asarray(adj)
        self.M = m

    def get_graph(self):
        G = nx.from_numpy_matrix(self.adj)

        return G

    def get_adj(self):

        return copy.deepcopy(self.__adj)

    def generate_B(self, half_gamma, x=None):
        """
        Generate B matrix from the graph: B_ij = a_ij * x_ij * gamma_i * gamma_j
        """

        M = self.M
        half_gamma = np.asarray(half_gamma)

        if half_gamma.shape[0] != M:
            raise ValueError('Input half_gamma vector should have compatible length {}'.format(self.M))
        if np.any(half_gamma==0):
            raise ValueError('Input half_gamma vector should not have zeros')

        # No additional edge activities
        if x is None:
            x = np.ones([M, M])
        else:
            x = np.atleast_1d(x)
            # Univariate edge activity
            if x.shape == (1,):
                x = np.ones([M, M]) * x
            # Multivariate edge activity, one for each edge
            else:
                m,n = x.shape
                if m != M or n != M:
                    raise ValueError('Input edge activities matrix should have compatible shape {}*{}'.format(self.M, self.M))

        B = self.__adj * x * half_gamma * half_gamma[:, np.newaxis]

        return B