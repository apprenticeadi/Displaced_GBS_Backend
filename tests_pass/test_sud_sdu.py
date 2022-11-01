import numpy as np
from scipy.stats import unitary_group

from src.gbs_experiment import PureGBS, sudGBS, sduGBS

from strawberryfields import Engine, Program
from strawberryfields.ops import S2gate

# Test that sud and sdu options in gbs_experiment.py are doing the right thing

M = 2
U = unitary_group.rvs(M)
r = np.random.uniform(low=0.5, high=2, size=(M, ))
alpha = np.random.uniform(low=1, high=3, size=(M,))

gbs = PureGBS(M)
gbs.add_squeezing(r)
gbs.add_interferometer(U)
gbs.add_displacement(alpha)
B = gbs.calc_B()
Gamma = gbs.calc_Gamma()
half_gamma = gbs.calc_half_gamma()
means, cov = gbs.state_xxpp()

gbs2 = sudGBS(M)
gbs2.add_squeezing(r)
gbs2.add_interferometer(U)
gbs2.add_displacement(alpha)
B2 = gbs2.calc_B()
Gamma2 = gbs2.calc_Gamma()
half_gamma2 = gbs2.calc_half_gamma()
means2, cov2 = gbs2.state_xxpp()

gbs3 = PureGBS(M)
gbs3.add_squeezing(r)
gbs3.add_displacement(alpha)
gbs3.add_interferometer(U)
B3 = gbs3.calc_B()
Gamma3 = gbs3.calc_Gamma()
half_gamma3 = gbs3.calc_half_gamma()
means3, cov3 = gbs3.state_xxpp()

gbs4 = sduGBS(M)
gbs4.add_squeezing(r)
gbs4.add_displacement(alpha)
gbs4.add_interferometer(U)
B4 = gbs4.calc_B()
Gamma4 = gbs4.calc_Gamma()
half_gamma4 = gbs4.calc_half_gamma()
means4, cov4 = gbs4.state_xxpp()

print(np.allclose(B, B2))
print(np.allclose(half_gamma, half_gamma2))
print(np.allclose(means, means2))
print(np.allclose(cov, cov2))

print(np.allclose(B3, B4))
print(np.allclose(half_gamma3, half_gamma4))
print(np.allclose(means3, means4))
print(np.allclose(cov3, cov4))

cov_Q = cov + np.identity(2* M) / 2
cov_Q_inv = np.linalg.inv(cov_Q)
exp_factor = -0.5 * means.T @ cov_Q_inv @ means

vacuum_prob = np.exp(exp_factor) / np.sqrt(np.linalg.det(cov_Q))

