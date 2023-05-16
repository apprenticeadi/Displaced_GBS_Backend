import numpy as np
from scipy.stats import unitary_group

from thewalrus import hafnian

from src.loop_hafnian_k_approx import loop_hafnian_approx_batch
from src.gbs_experiment import sduGBS
from src.utils import MatrixUtils


M = 16
r = 0.01
beta = 1
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

A = gbs.calc_A()
A_n = MatrixUtils.n_repetition(A, rpts)

Gamma = gbs.calc_Gamma()
Gamma_n = MatrixUtils.n_repetition(Gamma, rpts)

lhaf_approx = loop_hafnian_approx_batch(A_n, Gamma_n, approx=2)
lhaf_exact = hafnian(A_n + (Gamma_n - A_n.diagonal()) * np.eye(2*N), loop=True)

print(f'approx |lhaf|^2 = {np.absolute(lhaf_approx)**2}')
print(f'exact |lhaf|^2 = {np.absolute(lhaf_exact)**2}')