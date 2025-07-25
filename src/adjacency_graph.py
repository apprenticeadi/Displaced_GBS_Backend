import networkx as nx
import numpy as np
import copy


from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils
from src.gbs_matrix import GBSMatrix, GaussianMatrix

#TODO: Add vertex weighting to adjacency graph

class AdjacencyGraph:
    """
    Intended to be a wrapper class around networkx. Nx doesn't offer straightforward connection between graph object
    and its adjacency matrix. This class deals with graphs using solely its adjacency matrix, but can return a nx.Graph
    instance for visualisation and graph algorithms.
    Only takes in undirected graphs, with symmetric adjacency matrices.
    """

    def __init__(self, adj, vertex_weights=None):
        adj = np.asarray(adj)
        m, n = adj.shape

        if vertex_weights is None:
            vertex_weights = np.ones(m)
        vertex_weights = np.asarray(vertex_weights).flatten()
        l = vertex_weights.shape[0]

        if m != n:
            raise ValueError('Input adjacency matrix must be square matrix')
        if m != l:
            raise ValueError('Input adjacency matrix and vertex weights should have same dimensions')
        if not np.allclose(adj, adj.T):
            raise ValueError('Input adjacency matrix must be symmetric matrix')

        self.__adj = adj
        self.__vertex_weights = vertex_weights  # np array
        self.M = m

    def get_graph(self):
        G = nx.from_numpy_matrix(self.get_adj())
        for n in G.nodes:
            G.nodes[n]['weight'] = self.__vertex_weights[n]

        # self.__edge_weights = nx.get_edge_attributes(G, 'weight')  # dictionary

        for u, v, d in G.edges(data=True):
            weight = d['weight']
            distance = np.absolute(weight)
            G.edges[u, v]['distance'] = distance

        return G

    def get_adj(self):

        return copy.deepcopy(self.__adj)

    # Todo: draw loops
    def draw(self, show_edge_weights=False, node_size_min=100, node_size_max=1000, edge_width_min=2, edge_width_max=8,
             edge_transparency=0.5, edge_color='b', node_font_size=10, edge_font_size=10, ax_margins=0.08, ax_axis='off'):

        import matplotlib.pyplot as plt
        fig = plt.figure()

        G = self.get_graph()

        pos = nx.spring_layout(G, weight='distance', seed=7)  # positions for all nodes - seed for reproducibility

        # Node size scaled to vertex weights
        vw_min = min(self.__vertex_weights)
        vw_max = max(self.__vertex_weights)
        if np.isclose(vw_min, vw_max):
            node_sizes = 0.5 * (node_size_max + node_size_min) * np.ones(self.M)
        else:
            node_sizes = (self.__vertex_weights - vw_min) * (node_size_max - node_size_min) / (vw_max - vw_min) + node_size_min
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

        # Edge width scaled to edge weight magnitudes
        edge_dist_dict = nx.get_edge_attributes(G, 'distance')  # dictionary
        edge_dist = np.asarray(list(edge_dist_dict.values()))
        ed_min = min(edge_dist)
        ed_max = max(edge_dist)
        if np.isclose(ed_min, ed_max):
            edge_widths = 0.5 * (edge_width_max + edge_width_min) * np.ones(self.M)
        else:
            edge_widths = (edge_dist - ed_min) * (edge_width_max - edge_width_min) / (ed_max - ed_min) + edge_width_min
        nx.draw_networkx_edges(G, pos, edgelist=list(edge_dist_dict.keys()), width=edge_widths, alpha=edge_transparency, edge_color=edge_color)

        # Vertex labels
        vertex_labels = {u: f'{u}' for u in G.nodes()}
        nx.draw_networkx_labels(G, pos, vertex_labels, font_size=node_font_size)

        # Edge weight labels
        if show_edge_weights:
            edge_labels = {(u,v): f'{d["weight"]:.2f}' for u,v,d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=edge_font_size)

        ax = plt.gca()
        ax.margins(ax_margins)
        plt.axis(ax_axis)
        plt.tight_layout()

        plt.show()

class MatchingGraph(AdjacencyGraph):
    """
    Graph on which the matching polynomial is defined.
    """
    def __init__(self, adj, half_gamma=None, v=None, r_max=None, x=None):
        """
        Generate graph on which the matching polynomial is calculated

        :param adj: Adjacency of the graph
        :param half_gamma: Half gamma vector that fills B diagonal when calculating lHaf
        :param v: v vector that fills B diagonal when calculating the experiment setup. Default is zero vector if don't specify.
        :param r_max: The maximum squeezing paramter in the experiment
        :param x: Edge activity.
        """
        M = adj.shape[0]
        if not np.allclose(adj.diagonal(), np.zeros(M)):
            raise ValueError('Input adjacency matrix should not have loops')
        super().__init__(adj)


        self.set_half_gamma(half_gamma)
        if v is None:
            v = np.zeros(M)
        self.set_v(v)
        self.set_r_max(r_max)
        self.set_x(x)

    def set_half_gamma(self, half_gamma):
        if half_gamma is None:
            self.half_gamma = None
        else:
            half_gamma = np.asarray(half_gamma)
            if half_gamma.shape[0] != self.M:
                raise ValueError('Input half_gamma vector should have compatible length {}'.format(self.M))
            if np.any(half_gamma == 0):
                raise ValueError('Input half_gamma vector should not have zeros')
            self.half_gamma = half_gamma

    def set_v(self, v):
        v = np.asarray(v)
        if v.shape[0]!=self.M:
            raise ValueError('Input v vector for B diagonal should have compatible length {}'.format(self.M))
        else:
            self.v = v

    def set_r_max(self, r_max):
        if r_max is None:
            self.r_max = None
        else:
            if r_max.imag == 0:
                self.r_max = r_max
            else:
                raise ValueError('Input r_max must be real.')

    def set_x(self, x):
        M = self.M
        if x is None:
            self.x = None
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
                        'Multivariate edge activities matrix should have compatible shape {}*{}'.format(M, M))
                else:
                    self.x = x

    def generate_B(self):
        """
        Generate B matrix from the graph: B_ij = a_ij * x_ij * gamma_i * gamma_j
        """
        if self.x is None:
            raise AttributeError('Edge activity not defined')
        if self.half_gamma is None:
            raise AttributeError('Half gamma vector not defined')

        B = self.get_adj() * self.x * self.half_gamma * self.half_gamma[:, np.newaxis]
        B = MatrixUtils.filldiag(B, self.v)

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


