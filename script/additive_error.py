import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from src.photon_number_distributions import big_F


# matplotlib.rc('xtick', labelsize=18)
plt.rcParams.update({'font.size':14})
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

Ns = np.arange(start=2, stop=41, dtype=float)
w_labels = ['0', '0.1', '1', '2', 'N^0.5', 'N^1']
# w_labels=['0']
# w_labels = ['N^-1', '0.01', '0.1', '1', '2', '5', '10', 'N^1']

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

# plt.figure('bigF')
# for i in range(len(w_labels)):
#     plt.plot(Ns, bigFs[:, i], label=f'F(N, w={w_labels[i]})')
# plt.plot(Ns, np.power(2, Ns, dtype=np.float64), label='2^N')
# plt.plot(Ns, np.power(Ns**2, Ns) / factorial(Ns), label='M^N/N!')
# plt.yscale('log')
# plt.legend()

error_min = 1
error_max = 1

plt.figure('additive error', figsize=(8, 5))
for i in range(len(w_labels)):
    bigFs_i = bigFs[:,i]
    error = bigFs_i * factorial(Ns) / np.power(2*Ns**2, Ns)
    if np.min(error) < error_min:
        error_min = np.min(error)
    if np.max(error) > error_max:
        error_max = np.max(error)

    if w_labels[i]=='0':
        plt.plot(Ns, error, 'x', label=f'$w=${w_labels[i]}', color=cycle[i])
    elif w_labels[i][0]=='N':
        plt.plot(Ns, error, color=cycle[i])
        k = float(w_labels[i][2:])
        if k == 0.5:
            w_string = r'$w=\sqrt{N}$'
        elif k == 1:
            w_string = r'$w=N$'
        elif k == 0.25:
            w_string = r'$w=N^{1/4}$'
        plt.text(Ns[-1] * 0.9, error[-1] * 1.1, w_string, color=cycle[i])
    else:
        plt.plot(Ns, error, color=cycle[i])
        w_string = fr'$w=${w_labels[i]}'
        if float(w_labels[i]) < 5:
            plt.text(Ns[-1] * 0.9, error[-1]*1e10, w_string, color=cycle[i])
        else:
            plt.text(Ns[-1] * 0.9, error[-1] * 1e5, w_string, color=cycle[i])

plt.ylim([1e-70, 1e70])
plt.yscale('log')
# plt.legend(loc='lower left')
plt.xlabel(r'$N$')
plt.ylabel(r'additive error')
plt.yticks(np.logspace(-60, 60, num=7))