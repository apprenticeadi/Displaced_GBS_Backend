import numpy as np
from scipy.stats import unitary_group

from thewalrus import hafnian

from src.loop_hafnian_k_approx import loop_hafnian_approx_batch
from src.gbs_experiment import sduGBS
from src.utils import MatrixUtils


M = 16
r = 0.26
beta = 0.964
output_modes = [1,3,5,6]

N = int(np.sqrt(M))
K = N
rs = np.concatenate([r*np.ones(K), np.zeros(M-K)])
betas = np.concatenate([beta*np.ones(K), np.zeros(M-K)])
U = unitary_group.rvs(M)
output_n = np.zeros(M, dtype=int)
output_n[output_modes] = 1
rpts = np.concatenate([output_n, output_n])

gbs = sduGBS(M=M)
gbs.add_squeezing(rs)
gbs.add_displacement(betas)
gbs.add_interferometer(U)

B = gbs.calc_B()
B_n = MatrixUtils.n_repetition(B, output_n)
half_gamma = gbs.calc_half_gamma()
half_gamma_n = MatrixUtils.n_repetition(half_gamma, output_n)

lhaf_approx = loop_hafnian_approx_batch(B_n, half_gamma_n, approx=2)
lhaf_exact = hafnian(B_n + (half_gamma_n - B_n.diagonal()) * np.eye(N), loop=True)

print(f'approx |lhaf|^2 = {np.absolute(lhaf_approx)**2}')
print(f'exact |lhaf|^2 = {np.absolute(lhaf_exact)**2}')