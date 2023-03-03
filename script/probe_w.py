import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numba
import logging

from src.utils import LogUtils, DFUtils

@numba.njit(parallel=True)
def calc_Z_abs_values(N, num_trials, M):
    Z_abs_values = np.zeros(num_trials)
    for i in numba.prange(num_trials):
        X = np.random.normal(loc=0, scale=1/np.sqrt(2*M), size=N) + 1j * np.random.normal(loc=0, scale=1 / np.sqrt(2*M), size=N)
        Y = np.random.normal(loc=0, scale=1/np.sqrt(2*M), size=N) + 1j * np.random.normal(loc=0, scale=1 / np.sqrt(2*M), size=N)
        Z = np.sum(X * Y) / (np.sum(X) * np.sum(Y))
        Z_abs_values[i] = np.abs(Z)
    return Z_abs_values

# <<<<<<<<<<<<<<<<<<< Parameters   >>>>>>>>>>>>>>>>>>
n_start = 1
n_end = 6
num_n = 101
n_list = np.logspace(n_start, n_end, num=num_n, dtype=int)  # list of n values to test
num_trials = 100000  # number of trials to run for each n
stats = np.zeros((num_n, 4), dtype=float)  # Columns are n, min(|Z|), mean(|Z|), std(|Z|)
bins = np.logspace(-5, 1, num=100)

plot_fig = False
save_fig = False

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_w\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info(f'In this script, we find the scaling of |z| = |sum(x_i y_i)/(sum x_i)*(sum y_i)| for Gaussian i.i.d. '
             f'vectors x and y of length N, mean 0 and variance 1/N^2. '
             f'We calculate |z| for {num_n} Ns from N=1e{n_start} to 1e{n_end} and {num_trials} trials for each N')

if save_fig:
    plot_dir = dir + r'\plots'

if plot_fig:
    plt.figure('histograms')
for i, n in enumerate(n_list):
    M = n**2
    t0 = time.time()
    Z_abs_values = calc_Z_abs_values(n, num_trials, M)
    t1 = time.time()


    np.save(dir + fr'\N={n}.npy', Z_abs_values)
    t2 = time.time()

    min_Z_abs = np.min(Z_abs_values)
    mean_Z_abs = np.mean(Z_abs_values)
    std_Z_abs = np.std(Z_abs_values)

    stats[i] = np.array([n, min_Z_abs, mean_Z_abs, std_Z_abs])

    logging.info(f"N={n}, min={min_Z_abs}, mean={mean_Z_abs}, std={std_Z_abs}, time={t1-t0}, save_time = {t2-t1}")

    if plot_fig:
        if i  % 10 == 0:
            plt.hist(Z_abs_values, bins=bins, log=True, alpha=0.5, label=f'n={n}')
            plt.xscale('log')

if plot_fig:
    plt.legend()
    if save_fig:
        plt.savefig(DFUtils.create_filename(plot_dir + r'\histograms.png'))

np.save(dir + fr'\stats.npy', stats)


if plot_fig:
    # plot the mean values of |Z| for each value of n
    plt.figure('mean value of |Z|')
    plt.plot(stats[:, 0], stats[:, 2], label='mean(|Z|)')
    plt.plot(stats[:, 0], stats[:, 1], label='min(|z|)')
    plt.plot(stats[:, 0], 1 / stats[:, 0] ** 2, linestyle='-', color='black', label='1/M')
    plt.plot(stats[:, 0], 1 / stats[:, 0], linestyle='--', color='black', label='1/sqrt(M)')
    plt.plot(stats[:, 0], 1 / stats[:, 0] ** 4, linestyle='-.', color='black', label='1/M^2')
    plt.xlabel('N')
    plt.title('Mean and min of |Z| against N')
    # plt.xticks(n_list[::11])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    if save_fig:
        plt.savefig(DFUtils.create_filename(plot_dir + r'\mean_and_min.png'))

    # plot std of |Z|
    plt.figure('std of |Z|')
    plt.plot(stats[:, 0], stats[:, 3], label='std(|Z|)')
    plt.xlabel('N')
    plt.title('Std of |Z| against N')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(stats[:, 0], 1 / stats[:, 0], linestyle='--', color='black', label='1/sqrt(M)')
    plt.legend()
    if save_fig:
        plt.savefig(DFUtils.create_filename(plot_dir + r'\std.png'))