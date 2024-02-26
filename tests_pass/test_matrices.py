import numpy as np
from scipy.stats import unitary_group

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils

import strawberryfields as sf
from thewalrus.quantum import complex_to_real_displacements, Amat, probabilities

# Test gbs_experiment.py against strawberry fields

M = 16
K = 4
U = unitary_group.rvs(M)
r = np.random.random()
rs = np.concatenate([r * np.ones(K), np.zeros(M-K)])
beta = np.random.random() + 1j * np.random.random()
betas =  np.concatenate([beta * np.ones(K), np.zeros(M-K)])


print(U)
print(rs)
print(betas)

# Housemade functions
gbs = sduGBS(M)
gbs.add_squeezing(rs)
gbs.add_displacement(betas)
gbs.add_interferometer(U)
means, cov = gbs.state_xxpp()


# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, betas, rs, 'SDU', hbar=1)
sf_cov = sf_state.cov()
sf_means = sf_state.means()

print(np.allclose(sf_cov, cov))
print(np.allclose(sf_means, means))

# Compare A matrix and gamma vector
A = gbs.calc_A()
Gamma = gbs.calc_Gamma()

A_walrus = Amat(sf_cov, hbar=1)
alpha_walrus = complex_to_real_displacements(sf_means, hbar=1)  # this is (alpha, alpha.conj()) in our notation. alpha is the displacement in each mode after interferometer.
Gamma_walrus = alpha_walrus.conj() - A_walrus @ alpha_walrus

print(np.allclose(A, A_walrus))
print(np.allclose(Gamma, Gamma_walrus))


Gamma_simplified =  (beta.conjugate() - np.tanh(r) * beta) * np.sum(U[:, :K], axis=1)

print(np.allclose(Gamma_simplified, Gamma_walrus[:M]))