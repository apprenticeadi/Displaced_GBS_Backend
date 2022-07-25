import networkx as nx
import numpy as np
import copy

from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils
from src.gbs_matrix import GBSMatrix, GaussianMatrix

class AdjacencyGraph:
    """
    Intended to be a wrapper class around networkx. Nx doesn't offer straightforward connection between graph object
    and its adjacency matrix. This class deals with graphs using solely its adjacency matrix, but can return a nx.Graph
    instance for visualisation and graph algorithms.
    Only takes in undirected graphs, with symmetric adjacency matrices.
    """

    def __init__(self, adj):
        adj = np.asarray(adj)
        m, n = adj.shape
        if m != n:
            raise ValueError('Input adjacency matrix must be square matrix')

        if not np.allclose(adj, adj.T):
            raise ValueError('Input adjacency matrix must be symmetric matrix')

        self.__adj = np.asarray(adj)
        self.M = m

    def get_graph(self):
        G = nx.from_numpy_matrix(self.get_adj())

        return G

    def get_adj(self):

        return copy.deepcopy(self.__adj)


class UnloopedGraph(AdjacencyGraph):
    def __init__(self, adj):

        adj = MatrixUtils.filldiag(adj, np.zeros(adj.shape[0]))
        super().__init__(adj)


class MatchingGraph(UnloopedGraph):
    """
    Graph on which the matching polynomial is defined.
    """
    def __init__(self, adj, half_gamma, v, r_max, x=None):
        """
        Generate graph on which the matching polynomial is calculated

        :param adj: Adjacency of the graph
        :param half_gamma: Half gamma vector that fills B diagonal when calculating lHaf
        :param v: v vector that fills B diagonal when calculating the experiment setup
        :param r_max: The maximum squeezing paramter in the experiment
        :param x: Edge activity.
        """
        super().__init__(adj)
        M = self.M

        half_gamma = np.asarray(half_gamma)
        v = np.asarray(v)

        if half_gamma.shape[0] != M:
            raise ValueError('Input half_gamma vector should have compatible length {}'.format(self.M))
        if np.any(half_gamma == 0):
            raise ValueError('Input half_gamma vector should not have zeros')
        if v.shape[0]!=M:
            raise ValueError('Input v vector for B diagonal should have compatible length {}'.format(self.M))

        self.half_gamma = half_gamma
        self.v = v
        self.r_max = r_max
        # No additional edge activities
        if x is None:
            self.x = np.ones([M, M])
        else:
            x = np.atleast_1d(x)
            # Univariate edge activity
            if x.shape == (1,):
                self.x = np.ones([M, M]) * x
            # Multivariate edge activity, one for each edge
            else:
                m, n = x.shape
                if m != M or n != M:
                    raise ValueError(
                        'Input edge activities matrix should have compatible shape {}*{}'.format(self.M, self.M))
                else:
                    self.x = x

    def generate_B(self):
        """
        Generate B matrix from the graph: B_ij = a_ij * x_ij * gamma_i * gamma_j
        """

        B = self.__adj * self.x * self.half_gamma * self.half_gamma[:, np.newaxis]

        return B

    def generate_c(self):
        """
        Generates the real constant scaling factor c = tanh(rmax) / max|lambda^B_k|
        """

        B = self.generate_B()
        eigs_B = np.linalg.eigvalsh(B)
        c = np.tanh(self.r_max) / abs(max(eigs_B, key=abs))

        return c

    def generate_gbs_matrices(self):
        """
        Generate the GBS matrices rescaled by r_max.
        A = [[cB, 0], [0, cB^*]].
        Gamma = [sqrt(c)*halfgamma, sqrt(c)*halfgamma^*]

        :return: A, c*B, Gamma.
        """
        B = self.generate_B()
        c = self.generate_c()
        cB = c * B

        A = GBSMatrix.pure_A_from_B(cB)

        Gamma = np.sqrt(c) * np.concatenate([self.half_gamma, self.half_gamma.conjugate()])

        return A, cB, Gamma

    def generate_experiment(self):
        """
        Generates the experimental parameters: squeezing in each mode, displacement in each mode and interferometer
        unitary U
        :return: sq, displacement, U
        """

        A, cB, Gamma = self.generate_gbs_matrices()

        tanhr, U = takagi(cB)
        d_fock = GaussianMatrix.d_fock(A, Gamma)

        sq = np.arctanh(tanhr)
        displacement = d_fock[:self.M]

        return sq, displacement, U
