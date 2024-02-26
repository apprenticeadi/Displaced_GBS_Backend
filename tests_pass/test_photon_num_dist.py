import numpy as np
from src.photon_number_distributions import single_displaced_squeezed_vacuum, total_displaced_squeezed_vacuum
import random
import matplotlib.pyplot as plt

r = random.random() + 1j* random.random()
beta =  random.random() + 1j* random.random()
K = 3
# beta = 0
print(f'r={r}, beta={beta}')
N = np.sinh(np.abs(r))**2 + np.abs(beta)**2
print(f'mean photon number = {N}')
cutoff = K * int( N * 10) + 10
p_n = single_displaced_squeezed_vacuum(r, beta, cutoff)
n = np.arange(cutoff+1)

N_approx = np.sum(p_n * n)
print(f'approx mean photon number = {N_approx}')

plt.figure(0)
plt.bar(n, p_n)


p_total = total_displaced_squeezed_vacuum(r, beta, cutoff)

N_total_approx = np.sum(p_total * n)
N_total = K * N
print(f'total mean={N_total}, approx total mean = {N_total_approx}')

plt.figure(1)
plt.bar(n, p_total)