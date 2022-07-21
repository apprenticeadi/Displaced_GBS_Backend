import networkx as nx
import numpy as np
import copy

# This is intended to be a wrapper function around networkx, because I can't find a function that goes from nx.Graph to
# matrix.

class AjacencyMatrix:

    def __init__(self, adj):
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

    def generate_B_matrix(self, half_gamma, x=None):

        if half_gamma.shape[0] != self.M:
            raise ValueError('Input gamma vector should have compatible length {}'.format(self.M))

        if x is None:
            x = self.get_adj()
        else:
            m,n = x.shape
            if m != self.M or n != self.M:
                raise ValueError('Input edge activities matrix should have compatible shape {}*{}'.format(self.M, self.M))

        B = self.__adj * x * half_gamma * half_gamma[:, np.newaxis]

        return B