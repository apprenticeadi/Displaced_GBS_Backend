import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils


M = 30
delta = 20

x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

half_gamma = np.random.uniform(low=0.1, high=1, size=(M,))
v = np.random.uniform(low=0.01, high=1, size=(M,))  # I don't know what is a good starting point

B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + np.diag(v)

eigs_B_init = scipy.linalg.eigvalsh(B)  # When B is real, the square root of its eigenvalues are equal to its takagi spectrum
c_max = 1 / np.absolute(max(eigs_B_init, key=abs))

def get_alpha(half_gamma, B):
    Id = np.eye(M)

    inv_covQ = np.block([[Id, -B], [-B, Id]])
    covQ = scipy.linalg.inv(inv_covQ)

    Gamma = np.concatenate([half_gamma, half_gamma])  # Again keeping everything real for now

    d_fock = covQ @ Gamma
    alpha = d_fock[:M]
    alpha.sort()

    return alpha


max_scale = 100
num_points = 100

sq_phot = np.zeros(num_points)
dis_phot = np.zeros(num_points)
cs = np.zeros(num_points)

for i in range(num_points):

    c = i * c_max / max_scale
    cs[i] = c

    cB = c * B
    sqrtc_half_gamma = np.sqrt(c) * half_gamma

    tanhr, U = takagi(cB)
    sq = np.arctanh(tanhr)
    alpha = get_alpha(sqrtc_half_gamma, cB)

    sq_phot_vector = np.sinh(sq) ** 2
    sq_phot[i] = sum(sq_phot_vector)

    dis_phot_vector = np.absolute(alpha) ** 2
    dis_phot[i] = sum(dis_phot_vector)


plt.figure(0)
plt.plot(cs, sq_phot, label='sq phot')
plt.plot(cs, dis_phot, label='dis phot')
plt.xlabel('c')
plt.ylabel('Number of photons')
plt.title('M={},delta={}'.format(M, delta))
plt.legend()

ratio = sq_phot[1:] / dis_phot[1:]
plt.figure(1)
plt.plot(cs[1:], ratio)
plt.xlabel('c')
plt.ylabel('Sq/Dis ratio')
plt.title('M={},delta={}'.format(M, delta))