import numpy as np
import scipy
from scipy.optimize import minimize

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils

# A design protocol that can target arbitrary squeezing.
# Keep everything real in this script, so DBD is real symmetric/ Hermitian


M = 20
delta = 5

sq_dis_ratio = 100

# sq_target = np.random.uniform(low=0.5, high=1.5, size=(M,))
sq_phot_target = np.sqrt(M) * sq_dis_ratio / (1 + sq_dis_ratio)  # We want total photon number to be on order of sqrt(M)
sq_target = np.arcsinh(np.sqrt(sq_phot_target / M)) * np.ones(M)
sq_target.sort()
tanhr_target = np.tanh(sq_target)

print('sq_phot_target={}'.format(sq_phot_target))

dis_phot_target =  np.sqrt(M) / (1 + sq_dis_ratio) # np.random.uniform(low=30, high=120)

print('dis_phot_target={}'.format(dis_phot_target))

x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

half_gamma = np.ones(M)
v_init = sq_dis_ratio * np.ones(M)
B_init = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + v_init * np.eye(M)
eigs_B_init = scipy.linalg.eigvalsh(B_init)
d_init = np.sqrt(tanhr_target / abs(eigs_B_init))


def get_DBD(d, v):
    B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + np.diag(v)
    D = np.diag(d)
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

    alpha = get_alpha(d, v)
    dis_phot = sum(np.absolute(alpha) ** 2)

    if any(tanhr >= 1):
        return max(tanhr) ** 2 * M + dis_phot ** 2  # Penalise tanhr larger than 1
    elif any(alpha >=1):
        return dis_phot ** 2  # Penalise collisions by high displacements
    else:
        return sum(np.absolute(tanhr_target - tanhr) ** 2) / M + 0.01 * (dis_phot - dis_phot_target) ** 2 / M


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

sq_phot_vector = np.sinh(sq)**2
sq_phot = sum(sq_phot_vector)

dis_phot_vector = np.absolute(alpha)**2
dis_phot = sum(dis_phot_vector)

print('cost_sq={}'.format(sum(np.absolute(tanhr_target - tanhr) ** 2) / M))
print('cost_dis={}'.format((dis_phot - dis_phot_target) ** 2))

print('sq_phot={}'.format(sq_phot))
print('dis_phot={}'.format(dis_phot))

# print('sq_target={}'.format(sq_target))
# print('sq={}'.format(sq))
# print('alpha={}'.format(alpha))

mean_photon_vector = dis_phot_vector + np.diag(U @ np.diag(sq_phot_vector) @ U.T).real

print('mean_photon_vector={}'.format(mean_photon_vector))