import numpy as np
import matplotlib.pyplot as plt
from src.utils import DFUtils
import math


time_stamp = r'\17-02-2023(16-37-29.129467)'

dir = r'..\Results\probe_x' + time_stamp
raw_dir = dir + r'\raw'
save_fig = False
Ms = np.arange(start=10, stop=650, step=10)

# Each row contains [M, min of x_abs, error, max of x_abs, error, min of sum_x_abs, error, max of sum_x_abs, error]
data = np.zeros((Ms.shape[0], 9), dtype=float)

for iter, M in enumerate(Ms):

    # [min of x_abs, max of x_abs, min of sum_x_abs, max of sum_x_abs]
    data_M_i = np.load(raw_dir + fr'\M={M}_mins_and_maxes.npy')

    means = np.mean(data_M_i, axis=0)  # Takes mean along column
    stds = np.std(data_M_i, axis=0)  # Takes std along column

    data_M = np.array([means, stds]).T.flatten()

    data[iter, 0] = M
    data[iter, 1:] = data_M

np.save(dir + fr'\x_min_and_max_against_M.npy', data)

plt.figure(1)
plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], label='min(|x|)')
plt.plot(data[:, 0], data[:, 3], label='max(|x|)')
plt.plot(data[:, 0], data[:, 5], label='min(sum|x|)')
plt.plot(data[:, 0], data[:, 7], label='max(sum|x|)')

plt.plot(Ms, 1 / Ms, label='1/M', linestyle='-', color='black')
plt.plot(Ms, 1 / Ms ** 2, label='1/M^2', linestyle='--', color='black')
plt.plot(Ms, 1 / np.sqrt(Ms), label='1/sqrt(M)', linestyle='-.', color='black')
plt.xscale('log')
plt.yscale('log')
plt.ylim(bottom=1e-7)
plt.xlabel('M')
plt.legend(loc='lower left', borderpad=0.1)
plt.title(f'M={Ms[0]} to {Ms[-1]}')

if save_fig:
    plt.savefig(dir + r'\Plot_x_data_against_M.png')

for j in range(data_M_i.shape[1]):
    data_M_i[:, j][::-1].sort()
repeats = list(range(data_M_i.shape[0]))

plt.figure(2)
plt.plot(repeats, data_M_i[:, 0], label='min(|x|)')
plt.plot(repeats, data_M_i[:, 1], label='max(|x|)')
plt.plot(repeats, data_M_i[:, 2], label='min(sum|x|)')
plt.plot(repeats, data_M_i[:, 3], label='max(sum|x|)')
plt.hlines(1/M, xmin=repeats[0], xmax=repeats[-1], label='1/M', linestyle='-', color='black')
plt.hlines(1/M**2, xmin=repeats[0], xmax=repeats[-1], label='1/M^2', linestyle='--', color='black')
plt.hlines(1/np.sqrt(M), xmin=repeats[0], xmax=repeats[-1], label='1/sqrt(M)', linestyle='-.', color='black')
plt.xticks([1, repeats[-1]])
plt.legend(loc='upper right', borderpad=0.1)
plt.yscale('log')
plt.ylim(top=1e5)
plt.title(f'M={M}')

if save_fig:
    plt.savefig(dir + fr'\Plot_x_for_M={M}.png')
