import numpy as np
from scipy.stats import unitary_group
import itertools
from scipy.special import factorial
import random

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils, MatrixUtils
from thewalrus.quantum import density_matrix_element

import strawberryfields as sf
from thewalrus import hafnian

# Test probability calculations against thewalrus

M = 16
K = 4
U = unitary_group.rvs(M)
r = random.random()
rs = np.concatenate([r * np.ones(K), np.zeros(M-K)])
beta = random.random() + 1j * random.random()
betas = np.concatenate([beta *np.ones(K), np.zeros(M-K)])

outcome = np.concatenate([np.ones(K, dtype=int), np.zeros(M-K, dtype=int)])

N = np.sum(np.sinh(rs)**2) + np.sum(np.absolute(betas)**2)
print('Mean photon number = {}'.format(N))

# Housemade functions
gbs = sduGBS(M)
gbs.add_squeezing(rs)
gbs.add_displacement(betas)
gbs.add_interferometer(U)
means, cov = gbs.state_xxpp()


# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, betas, rs, 'SDU')
sf_means = sf_state.means()
sf_cov = sf_state.cov()

# Test Gaussian matrices
print('Test Gaussian matrices')
if not np.allclose(means, sf_means):
    raise Warning('Not consistent with sf for means')
if not np.allclose(cov, sf_cov):
    raise Warning('Not consistent with sf for cov')

# Test vacuum probability
print('Test vacuum probability')
vacuum_prob = gbs.vacuum_prob()
sf_vacuum_prob = sf_state.fock_prob([0]*M)
walrus_prob = density_matrix_element(means, cov, [0]*M, [0]*M, hbar=1)

if not np.isclose(vacuum_prob, sf_vacuum_prob):
    raise Warning('Not consistent with strawberryfields')
if not np.isclose(vacuum_prob, walrus_prob):
    raise Warning('Not consistent with walrus prob')




prob = gbs.prob(outcome)
sf_prob = sf_state.fock_prob(outcome, cutoff = 20)
walrus_prob = density_matrix_element(means, cov, list(outcome), list(outcome), hbar=1)  # outcome in thewalrus has to be a list
print(f'Test probability calculations for {outcome}')
print(np.allclose(prob, sf_prob))
print(np.allclose(prob, walrus_prob))
print(f'prob={prob}')