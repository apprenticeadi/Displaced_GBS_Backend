import numpy as np
import scipy
from scipy.optimize import minimize

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils, MatrixUtils
from src.adjacency_graph import MatchingGraph
from src.gbs_matrix import GaussianMatrix, GBSMatrix

# A design protocol that can target arbitrary squeezing.
# Keep everything real in this script, so DBD is real symmetric/ Hermitian


M = 20
delta = 5

sq_target = np.random.uniform(low=0.5, high=1.5, size=(M,))
sq_target.sort()
tanh_target = np.tanh(sq_target)


x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

half_gamma = np.random.uniform(low=0.1, high=1, size=(M,))  # + np.random.random(M) * 1j


v_init = np.asarray([1] * M)
B_init = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + v_init * np.eye(M)
eigs_B_init = scipy.linalg.eigvalsh(B_init)
d_init = np.sqrt(tanh_target / abs(max(eigs_B_init, key=abs)))


def get_DBD(d, v):
    B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + v * np.eye(M)
    D = d * np.eye(M)
    return D @ B @ D


def get_tanhr(d, v):
    DBD = get_DBD(d, v)
    # tanhr, _ = takagi(DBD)

    tanhr = np.absolute(scipy.linalg.eigvalsh(DBD))
    tanhr.sort()

    return tanhr


def get_alpha(d, v):
    DBD = get_DBD(d, v)
    Id = np.eye(M)

    inv_covQ = np.block([[Id, -DBD], [-DBD, Id]])
    covQ = scipy.linalg.inv(inv_covQ)

    Gamma = np.concatenate([d * half_gamma, d * half_gamma])  # Again keeping everything real for now

    d_fock = covQ @ Gamma
    alpha = d_fock[:M]
    alpha.sort()

    return alpha


def cost(dvg):
    d = dvg[:M]
    v = dvg[M:2 * M]

    tanhr = get_tanhr(d, v)

    if any(tanhr >= 1):
        return max(tanhr) ** 2  * M  # Penalise tanhr larger than 1
    else:
        return sum(np.absolute(tanh_target - tanhr) ** 2) / M


result = minimize(cost, x0=np.concatenate([d_init, v_init]))

print(result.success)

dvg_final = result.x
d_final = dvg_final[:M]
v_final = dvg_final[M:2 * M]

DBD_final = get_DBD(d_final, v_final)
tanhr, U = takagi(DBD_final)
tanhr.sort()

alpha = get_alpha(d_final, v_final)
sq = np.arctanh(tanhr)

# print(sq)
# print(alpha)


print(sum(np.sinh(sq) ** 2))
print(sum(np.absolute(alpha) ** 2))
