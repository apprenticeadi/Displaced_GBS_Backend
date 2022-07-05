import numpy as np

from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils
from src.random_graph import RandomGraph
from src.gbs_matrices import GraphMatrices, GaussianMatrices

M = 15
delta = 10  # maximum degree
x = -1  # edge activity
half_gamma = 1 - np.random.uniform(low=0, high=1, size=M)   # it cannot take 0 but can take 1
v =  np.ones(M) * 10000  # np.random.uniform(low=0, high=1, size=M) * 100000
r_max = 1
graph_G = RandomGraph(M=M, max_degree=delta)


Bmat = graph_G.generate_Bmatrix(x, half_gamma)
Bmat = MatrixUtils.filldiag(Bmat, v)

eigs_B = np.linalg.eigvalsh(Bmat)
c_factor = np.tanh(r_max) / abs(max(eigs_B, key=abs))
cB= c_factor * Bmat
gamma = np.sqrt(c_factor) * np.concatenate([half_gamma, half_gamma.conjugate()])

Amat = GraphMatrices.pure_A_from_B(cB)
mu_fock = GaussianMatrices.mu_fock_from_A(Amat, gamma)
tanhr, U = takagi(cB)

sq = np.arctanh(tanhr)
displacement = mu_fock[:M]

sq_photons = np.sum(np.sinh(sq)**2)
dis_photons = np.sum(displacement * displacement.conjugate()).real

print('sq: {}, dis: {}'.format(sq_photons, dis_photons))