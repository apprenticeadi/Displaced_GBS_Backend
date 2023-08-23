from thewalrus import hafnian
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from src.utils import MatrixUtils
import networkx as nx


N = 5
mean = 0
stddev = 1 / np.sqrt(2)

# X = np.arange(16, dtype=np.complex64).reshape((4,4))
X = np.random.normal(mean, stddev, (N,N)) + 1j * np.random.normal(mean, stddev, (N,N))
tildeX = (X @ X.T) / np.outer(np.sum(X, axis=1), np.sum(X, axis=1))

adjacency_matrix = MatrixUtils.filldiag(tildeX, np.ones(N))
lhaf_val = hafnian(adjacency_matrix, loop=True)
print(lhaf_val)

# Construct complete graph
G = nx.complete_graph(N)  # this shouldn't have self loops
assert len(list(nx.selfloop_edges(G))) == 0

# Construct Line graph
L = nx.line_graph(G)  # the graph, node and edge data are not propagated. But the nodes in L are now tuples of the form (u,v) for u,v in G

# Construct complement of line graph
LC = nx.complement(L)

coeffs = np.zeros(int(N/2) + 1, dtype=complex)  # highest degree is k=floor(N/2), and there is also zero-degree.
coeffs[0] = 1  # the polynomial zero-th degree coeff is 1, since product of diagonal terms is 1


# enumerate cliques in LC, does not include empty set.
for clique in nx.enumerate_all_cliques(LC):
    # each clique is a list of nodes in LC. they have the form [(u1,v1), (u2,v2), ..., (uk, vk)] for a k-clique.
    k = len(clique)
    c = 1.
    for node in clique:
        c = c * adjacency_matrix[node]

    coeffs[k] += c

matching_polynomial = Polynomial(coeffs)

mp_val = polyval(1, matching_polynomial.coef)
print(mp_val)
print(np.isclose(lhaf_val, mp_val))

roots = matching_polynomial.roots()


