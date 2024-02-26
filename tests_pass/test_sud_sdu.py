import numpy as np
from scipy.stats import unitary_group
import random

from src.gbs_experiment import PureGBS, sudGBS, sduGBS

from strawberryfields import Engine, Program
from strawberryfields.ops import S2gate

# Test that the different gbs models in gbs_experiment.py are internally consistent.

M = 16
K = 4  # number of input modes with squeezing and displacement
U = unitary_group.rvs(M)
r = random.random()
rs = np.concatenate([r * np.ones(K), np.zeros(M-K)])
alphas = np.random.uniform(low=0.5, high=1, size=(M,)) + 1j * np.random.uniform(low=0.5, high=1, size=(M,))
beta = (random.random() ) # + 1j * (random.random() + 1)
betas = np.concatenate([beta * np.ones(K), np.zeros(M-K)])

gbs = PureGBS(M)
gbs.add_squeezing(rs)
gbs.add_interferometer(U)
gbs.add_displacement(alphas)  # SUD order
B = gbs.calc_B()
Gamma = gbs.calc_Gamma()
half_gamma = gbs.calc_half_gamma()
means, cov = gbs.state_xxpp()
vac_prob = gbs.vacuum_prob()

gbs2 = sudGBS(M)
gbs2.add_squeezing(rs)
gbs2.add_interferometer(U)
gbs2.add_displacement(alphas)
B2 = gbs2.calc_B()
Gamma2 = gbs2.calc_Gamma()
half_gamma2 = gbs2.calc_half_gamma()
means2, cov2 = gbs2.state_xxpp()
vac_prob2 = gbs2.vacuum_prob()

gbs3 = PureGBS(M)
gbs3.add_squeezing(rs)
gbs3.add_displacement(betas)
gbs3.add_interferometer(U)  # SDU order
B3 = gbs3.calc_B()
Gamma3 = gbs3.calc_Gamma()
half_gamma3 = gbs3.calc_half_gamma()
means3, cov3 = gbs3.state_xxpp()
vac_prob3 = gbs3.vacuum_prob()

gbs4 = sduGBS(M)
gbs4.add_all(rs, betas, U)
B4 = gbs4.calc_B()
Gamma4 = gbs4.calc_Gamma()
half_gamma4 = gbs4.calc_half_gamma()
means4, cov4 = gbs4.state_xxpp()
vac_prob4 = gbs4.vacuum_prob()

print(np.allclose(B, B2))
print(np.allclose(half_gamma, half_gamma2))
print(np.allclose(means, means2))
print(np.allclose(cov, cov2))
print(np.allclose(vac_prob, vac_prob2))

print(np.allclose(B3, B4))
print(np.allclose(half_gamma3, half_gamma4))
print(np.allclose(means3, means4))
print(np.allclose(cov3, cov4))
print(np.allclose(vac_prob3, vac_prob4))

# The simplified vac prob expression for SDU

vac_prob5 = np.exp(- K * beta**2 * (1 - np.tanh(r))) / np.cosh(r) ** K
print(np.allclose(vac_prob3, vac_prob5))