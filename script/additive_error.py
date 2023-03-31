import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from src.photon_number_distributions import big_F

Ns = np.arange(start=2, stop=61, dtype=float)
w_labels = ['N^-1', 'N^0.25', '0.01', '0.1', '1', '2', '5', 'N^1']

bigFs = np.zeros((len(Ns), len(w_labels)), dtype=np.float64)
for i_N, N in enumerate(Ns):
    N = int(N)
    for i_w, w_label in enumerate(w_labels):

        if w_label[0] == 'N':
            k = float(w_label[2:])
            w = N**k

        else:
            w = float(w_label)

        bigFs[i_N, i_w] = big_F(w, N, N)[N]

plt.figure('bigF')
for i in range(len(w_labels)):
    plt.plot(Ns, bigFs[:, i], label=f'F(N, w={w_labels[i]})')
plt.plot(Ns, np.power(2, Ns, dtype=np.float64), label='2^N')
plt.plot(Ns, np.power(Ns**2, Ns) / factorial(Ns), label='M^N/N!')
plt.yscale('log')
plt.legend()

plt.figure('additive error')
for i in range(len(w_labels)):
    bigFs_i = bigFs[:,i]
    error = bigFs_i * factorial(Ns) / np.power(2*Ns**2, Ns)
    plt.plot(Ns, error, label=f'w={w_labels[i]}')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel(r'additive error')