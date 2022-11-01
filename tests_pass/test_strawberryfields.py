import numpy as np
from scipy.stats import unitary_group

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import TestUtils

import strawberryfields as sf

# Test that gbs_experiment.py against strawberry fields

M = 2
U = unitary_group.rvs(M)
r = np.random.uniform(low=0.5, high=2, size=(M, ))
alpha = np.random.uniform(low=1, high=3, size=(M,))

print(U)
print(r)
print(alpha)

# Housemade functions
gbs = PureGBS(M)
gbs.add_squeezing(r)
gbs.add_displacement(alpha)
gbs.add_interferometer(U)
means, cov = gbs.state_xxpp()


# Strawberryfields and thewalrus
sf_state = TestUtils.sf_circuit(M, U, alpha, r, 'SDU')
sf_cov = sf_state.cov()
sf_means = sf_state.means()

print(np.allclose(sf_cov, cov))
print(np.allclose(sf_means, means))

