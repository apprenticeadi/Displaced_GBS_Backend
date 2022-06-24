import numpy as np
import random

from strawberryfields.decompositions import williamson, bloch_messiah

from src.gbs_matrices import GraphMatrices, GaussianMatrices
from src.symplectic import Symplectic, SymplecticFock, SymplecticXXPP


M = 6  # Number of modes
max_degree = 4  # < M - 1
x = - 1 / 4*(max_degree - 1) - 1  # smaller than - 1 / 4*(max_degree - 1)

B_matrix = np.zeros([M,M])  # For now assume B matrix is real

# We can set the gamma vector to whatever we want and find the corresponding squeezing and unitary#
# BUT THIS IS NOT THE DISPLACEMENT VECTOR
half_gamma = np.ones(M)
gamma = np.concatenate([half_gamma, half_gamma.conjugate()])

node_dict = {k: [] for k in range(M)}  # keys are nodes in the graph, values are connected neighbours of the corresponding node
unfilled_nodes = list(range(M))
for node in unfilled_nodes:

    old_neighbours = node_dict[node]
    potential_neighbours = [a for a in unfilled_nodes if a not in old_neighbours and a!=node]

    if node == 0:
        num_new_neighbours = max_degree  # Set the degree of 0-th node as max_degree to ensure the max degree is achieved
        unfilled_nodes.pop(0)
    elif max_degree - len(old_neighbours) < len(potential_neighbours):
        # Pick a random number of new neighbours this node is allowed to have
        num_new_neighbours = random.randrange(0, max_degree - len(old_neighbours)+1)
    else:
        num_new_neighbours = random.randrange(0, len(potential_neighbours))

    # Randomly pick the new neighbours
    new_neighbours = random.sample(potential_neighbours, num_new_neighbours)

    for neighbour in new_neighbours:
        # Notify the new neighbours of this result
        node_dict[neighbour].append(node)

        # Remove nodes that have acheived maximum degree
        if len(node_dict[neighbour]) == max_degree:
            unfilled_nodes.remove(neighbour)
        elif len(node_dict[neighbour]) > max_degree:
            raise Exception('A node has larger degree than max degree')

        # Set the B matrix off diagonal value
        B_matrix[node, neighbour] = x * half_gamma[node] * half_gamma[neighbour]
        B_matrix[neighbour, node] = x * half_gamma[node] * half_gamma[neighbour]

    node_dict[node] = old_neighbours + new_neighbours

# c factor must be smaller than inverse of the absolute value of B eig of maximum abs value
eigs_B = np.linalg.eigvalsh(B_matrix)
c_factor = random.uniform(0.0, 1.0) * 1 / abs(max(eigs_B, key=abs))

# Construct the A matrix: A = [[cB, 0], [0, cB]]
O_mat = np.zeros_like(B_matrix)
A_mat = np.block([[c_factor*B_matrix, O_mat], [O_mat, c_factor*B_matrix]])

assert GraphMatrices.is_valid_Amat(A_mat) == True

Sigma_fock = GaussianMatrices.cov_fock_from_A(A_mat, dtype=np.float64)
Sigma_xxpp = Symplectic.matrix_fock_to_xxpp(Sigma_fock)  # The williamson function only works for real symmetric matrix

mu_fock = GaussianMatrices.mu_fock_from_A(A_mat, gamma, dtype=np.float64)
mu_xxpp = Symplectic.vector_fock_to_xxpp(mu_fock)

D, S = williamson(Sigma_xxpp)
assert np.allclose(D, 0.5*np.identity(2*M))

O1, smat, O2 = bloch_messiah(S)

squeezing = np.log(np.diagonal(smat)[:M])
displacement = mu_xxpp[:M] + mu_xxpp[M:]