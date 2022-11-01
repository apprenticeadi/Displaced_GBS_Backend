import numpy as np
from scipy.stats import unitary_group

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils

import strawberryfields as sf

# Test script for vacuum probability calculation for different models of gbs experiments

M = 2
U = unitary_group.rvs(M)
r = np.random.uniform(low=0.5, high=2, size=(M, ))
alpha = np.random.uniform(low=1, high=3, size=(M,))

# Housemade functions
gbs = PureGBS(M)
gbs.add_squeezing(r)
gbs.add_displacement(alpha)
gbs.add_interferometer(U)

# B = gbs.calc_B()
# Gamma = gbs.calc_Gamma()
# half_gamma = gbs.calc_half_gamma()
# means, cov = gbs.state_xxpp()

vacuum_prob = gbs.vacuum_prob()

gbs2 = sduGBS(M)
gbs2.add_squeezing(r)
gbs2.add_interferometer(U)
gbs2.add_displacement(alpha)
vacuum_prob2 = gbs2.vacuum_prob()

print(np.isclose(vacuum_prob, vacuum_prob2))

# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, alpha, r, 'SDU')
sf_vacuum_prob = sf_state.fock_prob([0]*M)

print(np.isclose(vacuum_prob, sf_vacuum_prob))