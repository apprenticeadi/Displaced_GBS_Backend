import numpy as np
import scipy
from scipy.optimize import minimize

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils

# A design protocol that tries to target arbitrary sq/dis ratio.
# Keep everything real in this script, so DBD is real symmetric/ Hermitian


M = 10
delta = 5

sq_dis_ratio = 0.0001

# sq_target = np.random.uniform(low=0.5, high=1.5, size=(M,))
sq_phot_target = np.sqrt(M) * sq_dis_ratio / (1 + sq_dis_ratio)  # We want total photon number to be on order of sqrt(M)
sq_target = np.arcsinh(np.sqrt(sq_phot_target / M)) * np.ones(M)
sq_target.sort()
tanhr_target = np.tanh(sq_target)

print('sq_phot_target={}'.format(sq_phot_target))

dis_phot_target = np.sqrt(M) / (1 + sq_dis_ratio)  # np.random.uniform(low=30, high=120)

print('dis_phot_target={}'.format(dis_phot_target))

x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

half_gamma_init = np.ones(M)
v_init = half_gamma_init


def get_B(half_gamma, v):
    B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + np.diag(v)
    return B


B_init = x * np.multiply(adj * half_gamma_init, half_gamma_init[:, np.newaxis]) + v_init * np.eye(M)
eigs_B_init = scipy.linalg.eigvalsh(B_init)
d_max = np.sqrt(max(tanhr_target) / np.absolute(max(eigs_B_init, key=abs)))  # keep it the same, since target sq is the same across all modes
d_vary = np.sqrt(tanhr_target / np.absolute(eigs_B_init))

half_gamma_init = d_max * half_gamma_init
v_init = d_max ** 2 * v_init


def get_tanhr(half_gamma, v):
    B = get_B(half_gamma, v)
    # tanhr, _ = takagi(DBD)

    tanhr = np.absolute(scipy.linalg.eigvalsh(B))
    tanhr.sort()

    return tanhr


def get_alpha(half_gamma, v):
    B = get_B(half_gamma, v)
    Id = np.eye(M)

    inv_covQ = np.block([[Id, -B], [-B, Id]])
    covQ = scipy.linalg.inv(inv_covQ)

    Gamma = np.concatenate([half_gamma, half_gamma])  # Again keeping everything real for now

    d_fock = covQ @ Gamma
    alpha = d_fock[:M]
    alpha.sort()

    return alpha


def cost(gamma_v):
    half_gamma = gamma_v[:M]
    v = gamma_v[M:2*M]

    tanhr = get_tanhr(half_gamma, v)

    alpha = get_alpha(half_gamma, v)
    dis_phot = sum(np.absolute(alpha) ** 2)

    coeff = 0.01 * np.average(tanhr) / dis_phot

    if any(tanhr >= 1):
        return max(tanhr) ** 2 + coeff * dis_phot_target ** 2 # Penalise tanhr larger than 1
    elif any(alpha >= 1):
        return max(tanhr) ** 2 + coeff * dis_phot_target ** 2   # Penalise collisions caused by high collisions
    else:
        return sum(np.absolute(tanhr_target - tanhr) ** 2) / M + coeff * (dis_phot - dis_phot_target) ** 2 / M


result = minimize(cost, x0=np.concatenate([half_gamma_init, v_init]))

print(result.success)

gamma_v_final = result.x
half_gamma_final = gamma_v_final[:M]
v_final = gamma_v_final[M:2 * M]

B_final = get_B(half_gamma_final, v_final)
tanhr, U = takagi(B_final)
tanhr.sort()

alpha = get_alpha(half_gamma_final, v_final)
sq = np.arctanh(tanhr)

sq_phot_vector = np.sinh(sq) ** 2
sq_phot = sum(sq_phot_vector)

dis_phot_vector = np.absolute(alpha) ** 2
dis_phot = sum(dis_phot_vector)

print('cost_sq={}'.format(sum(np.absolute(tanhr_target - tanhr) ** 2) / M))
print('cost_dis={}'.format((dis_phot - dis_phot_target) ** 2))

print('sq_phot={}'.format(sq_phot))
print('dis_phot={}'.format(dis_phot))

# print('sq_target={}'.format(sq_target))
# print('sq={}'.format(sq))
# print('alpha={}'.format(alpha))

mean_photon_vector = dis_phot_vector + np.sum(np.square(np.absolute(U)) * sq_phot_vector, axis=1)

print('mean_photon_vector={}'.format(mean_photon_vector))