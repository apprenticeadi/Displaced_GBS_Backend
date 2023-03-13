import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
import os

from src.utils import DFUtils
from src.photon_number_distributions import big_F


def prefactor(N, func):
    if func == 'det' or func == 'perm':
        return 1 / np.sqrt(factorial(N))
    elif func == 'haf':
        if N % 2 ==0 :
            return np.sqrt(comb(N**2 + N - 1, N) / comb(N-1, N/2) / np.power(float(N), 2.*N))
        else:
            return 0

    elif func[:4] == 'lhaf':
        w = float(func[func.index('w')+2:])
        return np.sqrt(comb(N**2 + N-1, N) * np.power(2./N**2, float(N)) / big_F(w, N, N)[N])

    else:
        raise ValueError('func not recognized')



repeat_i = 1000  #number of data in each raw data file
repeats=100000

dir_head = fr'..\Results\anticoncentration_over_X\{repeats}repeats'
func_dirs = os.listdir(dir_head)

func = 'perm'
Ns = np.arange(start=6, stop=26, step=2)

filtered_dir = [file_ for file_ in func_dirs if file_.startswith(func)][0]  # For now just take the first directory, but later need to modify this when data become more complicated

raw_data_arr = np.zeros((len(Ns), repeats), dtype=float)
refactored_data_arr = np.zeros((len(Ns), repeats), dtype=float)
plt.figure(f'{func}')
for j, N in enumerate(Ns):
    N_dir = dir_head + fr'\{filtered_dir}\N={N}'

    raw_data_files = os.listdir(N_dir)

    combined_raw_data = np.zeros((len(raw_data_files), repeat_i), dtype=float)
    for i, fn in enumerate(raw_data_files):
        combined_raw_data[i, :] = np.load(N_dir + fr'\{fn}')
    combined_raw_data = combined_raw_data.flatten()
    combined_raw_data.sort()


    raw_data_arr[j, :] = combined_raw_data

    pf = prefactor(N, func)
    refactored_data = pf * combined_raw_data
    refactored_data_arr[j, :] = refactored_data

    plt.plot(list(range(repeats)), refactored_data, label=f'N={N}')

plt.legend()
plt.yscale('log')
plt.xticks([1, repeats])
plt.title(func)


# Find cumulative distribution function

x_min = np.min(refactored_data_arr)
x_min_log = np.floor(np.log10(x_min))  # the minimum value much more likely to come from lhaf with w=1
xs = np.logspace(start=-2, stop=0, num=8)

# first row is xs values, second to last row is cumulative distribution for each N
cum_distrib = np.zeros((len(Ns) + 1, len(xs)), dtype=float)
cum_distrib[0, :] = xs
for i in range(refactored_data_arr.shape[0]):

    for j, x in enumerate(xs):
        cum_distrib[i+1, j] = np.argmax(refactored_data_arr[i] > x) / repeats

plt.figure(f'{func} F(x,N)')
for i, x in enumerate(xs):
    plt.plot(Ns, cum_distrib[1:, i],  label=f'x={x:.3}')

plt.legend()
plt.xlabel('N')
plt.yscale('log')
plt.xscale('log')
# plt.xticks(Ns)