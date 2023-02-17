import numpy as np
from scipy.stats import unitary_group
import time

from src.ACC_functions import lhaf_squared
from src.gbs_experiment import sduGBS

from thewalrus import hafnian

N = 32  # 32 is the limit on tabletop computer without parallelization
M = N ** 2

outcome = np.zeros(M, dtype=int)
outcome[:N] = 1
loop = True

beta = np.random.random() + 1
r = np.random.random() + 1
w = beta * (1-np.tanh(r)) / np.sqrt(np.tanh(r))

print(f'beta={beta}, r={r}, w={w}')


betas = beta * np.ones(M)
rs = r * np.ones(M)

U = unitary_group.rvs(M)

gbs_experiment = sduGBS(M)
gbs_experiment.add_all(rs, betas, U)

time0 = time.time()
lhaf_0 = gbs_experiment.lhaf(outcome, displacement=loop)
time1 = time.time()

lhaf2_0 = np.absolute(lhaf_0) ** 2

print(f'lhaf2_0 = {lhaf2_0:.3} for time={time1-time0}')
# The time here is slower than calling hafnian second time.
# there must be some overhead in calling hafnian() function for the first time.

time2 = time.time()
lhaf2_1 = lhaf_squared(U, w, N, loop=loop) * np.tanh(r)**N
time3 = time.time()

print(f'lhaf2_1 = {lhaf2_1:.3} for time={time3-time2}')

print(f'Same result = {np.isclose(lhaf2_0, lhaf2_1)}')


B = gbs_experiment.calc_B()
B1 = np.tanh(r) * U@U.T
print(f'Correct B ={np.allclose(B, B1)}')

gamma = gbs_experiment.calc_half_gamma()
gamma1 = beta * (1-np.tanh(r)) * np.sum(U, axis=1)
print(f'Correct gamma ={np.allclose(gamma, gamma1)}')



