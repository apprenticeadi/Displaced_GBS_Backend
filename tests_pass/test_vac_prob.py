import numpy as np
from scipy.stats import unitary_group
import itertools
from math import factorial

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils, MatrixUtils
from src.thewalrus_quantum import prefactor, density_matrix_element

import strawberryfields as sf
from thewalrus import hafnian

# Test script for vacuum probability calculation for different models of gbs experiments

M = 2
U = unitary_group.rvs(M)
r = np.random.uniform(low=0.5, high=1, size=(M, ))
alpha = np.random.uniform(low=0.5, high=1, size=(M,))
# alpha = np.zeros(M)

# Housemade functions
gbs = PureGBS(M)
gbs.add_squeezing(r)
gbs.add_interferometer(U)
gbs.add_displacement(alpha)
means, cov = gbs.state_xxpp()

gbs2 = sudGBS(M)
gbs2.add_squeezing(r)
gbs2.add_interferometer(U)
gbs2.add_displacement(alpha)

# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, alpha, r, 'SUD')
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
vacuum_prob2 = gbs2.vacuum_prob()
sf_vacuum_prob = sf_state.fock_prob([0]*M)
walrus_prefactor = prefactor(means, cov, hbar=1)
if not np.isclose(vacuum_prob, vacuum_prob2):
    raise Warning('Not self consistent')
if not np.isclose(vacuum_prob, sf_vacuum_prob):
    raise Warning('Not consistent with strawberryfields')
if not np.isclose(vacuum_prob, walrus_prefactor):
    raise Warning('Not consistent with walrus prefactor')


# Test GBS matrices
print('Test GBS matrices')
B = gbs.calc_B()
half_gamma = gbs.calc_half_gamma()
B2 = gbs2.calc_B()
half_gamma2 = gbs2.calc_half_gamma()
if not np.allclose(B,B2):
    raise Warning('Not self consistent for B matrix')
if not np.allclose(half_gamma, half_gamma2):
    raise Warning('Not self consistent for half_gamma')


def gbs_prob(outcome):
    B_n = MatrixUtils.n_repetition(B, outcome)
    half_gamma_n = MatrixUtils.n_repetition(half_gamma, outcome)
    haf_B = MatrixUtils.filldiag(B_n, half_gamma_n)
    prob = vacuum_prob * np.absolute(hafnian(haf_B, loop=True)) ** 2
    for n in outcome:
        prob /= factorial(n)
    return prob


def test_prob(outcome):
    prob=gbs_prob(outcome)
    sf_prob = sf_state.fock_prob(outcome)
    print('For {}, prob={}'.format(outcome, prob))
    if not np.isclose(prob, sf_prob):
        raise Warning('Not consistent with strawberryfields for {}'.format(outcome))


# Test probabilities
# TODO: This test never succeeds for displaced circuits.
outcomes = list(itertools.product([0,1], repeat=M))
for ns in outcomes:
    test_prob(ns)

