import numpy as np
import scipy
from scipy.optimize import minimize

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils

# A design protocol that can target arbitrary squeezing.
# Keep everything real in this script, so DBD is real symmetric/ Hermitian


M = 10
delta = 5

n_phot = np.sqrt(M)  # This will likely keep experiment in collisionless regime. Later will verify with mean_photon_vector.
sq_max = np.arcsinh(np.sqrt(n_phot / M))

sq_target = np.random.uniform(low=sq_max / 10, high=sq_max, size=(M,))  # Can set this to be arbitrary
sq_target.sort()
tanhr_target = np.tanh(sq_target)
n_sq_target = sum(np.sinh(sq_target)**2)

print('n_sq_target={}'.format(n_sq_target))

n_dis_target = n_phot - n_sq_target

print('n_dis_target={}'.format(n_dis_target))

x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

half_gamma_init = np.ones(M)
v_init = np.ones(M)  # I don't know what is a good starting point


def get_B(half_gamma, v):
    B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + np.diag(v)
    return B


B_init = x * np.multiply(adj * half_gamma_init, half_gamma_init[:, np.newaxis]) + v_init * np.eye(M)
eigs_B_init = scipy.linalg.eigvalsh(B_init)
d_max = np.sqrt(max(tanhr_target) / np.absolute(max(eigs_B_init, key=abs)))
# d_vary = np.sqrt(tanhr_target / np.absolute(eigs_B_init))

# Scale the inital half_gamma and v, so the optimisation starts with a physically valid B matrix.
half_gamma_init = d_max * half_gamma_init  # Should I use d_max or d_vary I don't know.
v_init = d_max ** 2 * v_init


def get_tanhr(half_gamma, v):
    B = get_B(half_gamma, v)
    # tanhr, _ = takagi(DBD)

    tanhr = np.absolute(scipy.linalg.eigvalsh(B))  # When B is real, takagi spectrum equals the magnitude of eigval(B).
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
        return max(tanhr) ** 2 + coeff * n_dis_target ** 2 # Penalise tanhr larger than 1
    elif any(alpha >= 1):
        return max(tanhr) ** 2 + coeff * n_dis_target ** 2   # Penalise collisions caused by high collisions
    else:
        return sum(np.absolute(tanhr_target - tanhr) ** 2) / M + coeff * (dis_phot - n_dis_target) ** 2 / M


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
print('cost_dis={}'.format((dis_phot - n_dis_target) ** 2))

print('sq_phot={}'.format(sq_phot))
print('dis_phot={}'.format(dis_phot))

# print('sq_target={}'.format(sq_target))
# print('sq={}'.format(sq))
# print('alpha={}'.format(alpha))

mean_photon_vector = dis_phot_vector + np.sum(np.square(np.absolute(U)) * sq_phot_vector, axis=1)

print('mean_photon_vector={}'.format(mean_photon_vector))
