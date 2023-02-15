import numpy as np
from scipy.stats import unitary_group

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils

import strawberryfields as sf
from thewalrus.quantum import complex_to_real_displacements, Amat, probabilities

# Test that gbs_experiment.py against strawberry fields

M = 10
U = unitary_group.rvs(M)
r = np.random.uniform(low=0.5, high=2, size=(M, ))
alpha = np.random.uniform(low=1, high=3, size=(M,))

print(U)
print(r)
print(alpha)

# Housemade functions
gbs = sduGBS(M)
gbs.add_squeezing(r)
gbs.add_displacement(alpha)
gbs.add_interferometer(U)
means, cov = gbs.state_xxpp()


# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, alpha, r, 'SDU', hbar=1)
sf_cov = sf_state.cov()
sf_means = sf_state.means()

print(np.allclose(sf_cov, cov))
print(np.allclose(sf_means, means))

# Compare A matrix and gamma vector
A = gbs.calc_A()
Gamma = gbs.calc_Gamma()

A_walrus = Amat(sf_cov, hbar=1)
beta_walrus = complex_to_real_displacements(sf_means, hbar=1)  # this is (alpha, alpha.conj()) in our notation. alpha is the displacement in each mode after interferometer.
Gamma_walrus = beta_walrus.conj() - A_walrus @ beta_walrus

print(np.allclose(A, A_walrus))
print(np.allclose(Gamma, Gamma_walrus))
