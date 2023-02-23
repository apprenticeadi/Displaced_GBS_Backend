import time

import numpy as np
from scipy.stats import unitary_group

import logging
import datetime
import matplotlib.pyplot as plt

from src.utils import MatrixUtils, LogUtils, DFUtils

# <<<<<<<<<<<<<<<<<<< Parameters  >>>>>>>>>>>>>>>>>>
Ms = np.arange(start=10, stop=210, step=10)
M_num = len(Ms)
repeat = 1000  # For each M, we average over this many Haar random unitaries
w = 1  # The diagonal weight.

save_fig = False
save_raw_x = True
if repeat > 1:
    save_raw_x = False

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_x\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('In this script, we find the scaling of |x_ij|=|B_ij/gamma_i*gamma_j| with M for '
             'diagonal weight w = beta*(1-tanh(r))/sqrt(tanhr) = {}.'.format(w))

# Each row contains [M, min of x_abs, error, max of x_abs, error, min of sum_x_abs, error, max of sum_x_abs, error]
data = np.zeros((M_num, 9), dtype=float)

for iter, M in enumerate(Ms):

    N = int(np.floor(np.sqrt(M)))
    # [min of x_abs, max of x_abs, min of sum_x_abs, max of sum_x_abs]
    data_M_i = np.zeros((repeat, 4))

    for i in range(repeat):
        time0 = time.time()
        U = unitary_group.rvs(M)  # this is the most costly step

        B = U @ U.T  # The tanhr term is absorbed inside w
        half_gamma = w * np.sum(U, axis=1)
        x = B / np.outer(half_gamma, half_gamma)
        np.fill_diagonal(x, 0)  # We don't want x_ii

        if save_raw_x:
            np.save(DFUtils.create_filename(dir + fr'\raw\M={M}\raw_x_{i}.npy'), x)

        x_n = x[:N, :N]  # we only want the top left N times N submatrix
        x_abs = np.absolute(x)
        masked_x_abs = x_abs[
            ~np.eye(len(x_abs), dtype=bool)]  # mask out diagonal terms which are zero. masked shaped is 1d
        sum_x_abs = np.sum(x_abs[:N,:N], axis=1)

        data_M_i[i] = np.array([np.min(masked_x_abs), np.max(masked_x_abs), np.min(sum_x_abs), np.max(sum_x_abs)])

        time_final = time.time()
        print(
            f'time={time_final - time0:.3}')

    np.save(DFUtils.create_filename(dir + fr'\raw\M={M}_mins_and_maxes.npy'), data_M_i)

    means = np.mean(data_M_i, axis=0)  # Takes mean along column
    stds = np.std(data_M_i, axis=0)  # Takes std along column

    data_M = np.array([means, stds]).T.flatten()

    logging.info('for M={}, '
                 '[min of x_abs, error, max of x_abs, error, min of sum_x_abs, error, max of sum_x_abs, error] is {}'
                 .format(M, data_M))

    data[iter, 0] = M
    data[iter, 1:] = data_M

np.save(dir + fr'\x_min_and_max_against_M.npy', data)

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>

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
plt.xlabel('M')
plt.legend()

plt.ylim(bottom=1e-7)

# plt.figure(2)
# plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], label='min(|x|)')
# plt.errorbar(data[:, 0], data[:, 3], yerr=data[:, 4], label='max(|x|)')
# plt.errorbar(data[:, 0], data[:, 5], yerr=data[:, 6], label='min(sum|x|)')
# plt.errorbar(data[:, 0], data[:, 7], yerr=data[:, 8], label='max(sum|x|)')
#
# plt.plot(Ms, 1 / Ms, label='1/M', linestyle='-', color='black')
# plt.plot(Ms, 1 / Ms ** 2, label='1/M^2', linestyle='--', color='black')
# plt.plot(Ms, 1 / np.sqrt(Ms), label='1/sqrt(M)', linestyle='-.', color='black')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('M')
# plt.legend()

if save_fig:
    plt.savefig(dir + r'\Plot_x_data_against_M.png')
