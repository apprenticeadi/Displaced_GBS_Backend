import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt
import random

from thewalrus import hafnian

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import LogUtils, DFUtils, MatrixUtils


# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
M = 50
N = 4
w = 1  # the weight on the diagonal entries
repeat = 10000

# This script sets up an experiment and calculates the probabilities
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = fr'..\Results\n_vs_U\M={M}_N={N}_{time_stamp}'
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Calculate the lhaf of interest for two cases. Case (a), generate a single Haar random unitary, and '
             'calculate lhaf^2 for different outcomes n. Case (b), fix outcome n to be top left N*N submatrix, and '
             'calculate lhaf^2 for different Haar random U. Numerically see if the distribution of the lhaf^2 values are'
             'similar. M={}, N={}, diagonal weight w={}, repeat={}'.format(M, N, w, repeat))

# <<<<<<<<<<<<<<<<<<< Optional parameters  >>>>>>>>>>>>>>>>>>
save_U = False
plotting = True
if plotting:
    plot_dir = dir + r'\plots'

# <<<<<<<<<<<<<<<<<<< Function for calculating lhaf  >>>>>>>>>>>>>>>>>>
def loop_hafnian_squared(B, gamma, outcome, print_msg=True, loop=True):
    B_n = MatrixUtils.n_repetition(B, outcome)
    gamma_n = MatrixUtils.n_repetition(gamma, outcome)
    haf_B = MatrixUtils.filldiag(B_n, gamma_n)

    time1 = time.time()
    lhaf2 = np.absolute(hafnian(haf_B, loop=loop)) ** 2
    time2 = time.time()

    if print_msg:
        print(f'Time ={time2 - time1}, lhaf2={lhaf2}, click modes={np.where(outcome==1)}')

    return lhaf2

# <<<<<<<<<<<<<<<<<<< Same U, different n  >>>>>>>>>>>>>>>>>>
U0 = unitary_group.rvs(M)
B0 = U0 @ U0.T
gamma0 = w * np.sum(U0, axis=1)
if save_U:
    np.save(DFUtils.create_filename(dir + rf'\U_0.npy'), U0)

lhaf2s_n = np.zeros(repeat, dtype=float)
time_initial = time.time()
for iter in range(repeat):
    click_modes = random.sample(range(M), N)  # These are the modes that click. We only care about collisionless outcomes.
    outcome_n = np.zeros(M, dtype=int)
    for mode in click_modes:
        outcome_n[mode] = 1

    lhaf2 = loop_hafnian_squared(B0, gamma0, outcome_n)

    lhaf2s_n[iter] = lhaf2

time_final = time.time()
np.save(DFUtils.create_filename(dir + r'\lhaf2_over_n.npy'), lhaf2s_n)
logging.info(f'Time={time_final-time_initial} to calculate {repeat} {N}*{N} loop Hafnian squared of interest for fixed U random n')

# <<<<<<<<<<<<<<<<<<< Same n, different U  >>>>>>>>>>>>>>>>>>
outcome_U = np.zeros(M, dtype=int)
outcome_U[:N] = 1
lhaf2s_U = np.zeros(repeat, dtype=float)
time_initial = time.time()
for iter in range(repeat):

    U = unitary_group.rvs(M)
    if save_U:
        np.save(DFUtils.create_filename(dir + rf'U_{iter+1}.npy'), U)


    B = U @ U.T
    gamma = w * np.sum(U, axis=1)

    lhaf2 = loop_hafnian_squared(B, gamma, outcome_U)

    lhaf2s_U[iter] = lhaf2

time_final=time.time()
np.save(DFUtils.create_filename(dir+r'\lhaf2_over_U.npy'), lhaf2s_U)
logging.info(f'Time={time_final-time_initial} to calculate {repeat} {N}*{N} loop Hafnian squared of interest for fixed n random U')

# <<<<<<<<<<<<<<<<<<< Same n, different U, haf^2  >>>>>>>>>>>>>>>>>>
outcome_U = np.zeros(M, dtype=int)
outcome_U[:N] = 1
haf2s_U = np.zeros(repeat, dtype=float)
time_initial = time.time()
for iter in range(repeat):

    U = unitary_group.rvs(M)
    if save_U:
        np.save(DFUtils.create_filename(dir + rf'U_{iter+1}.npy'), U)


    B = U @ U.T
    gamma = w * np.sum(U, axis=1)

    haf2 = loop_hafnian_squared(B, gamma, outcome_U, loop=False)

    haf2s_U[iter] = haf2

time_final=time.time()
np.save(DFUtils.create_filename(dir+r'\haf2_over_U.npy'), haf2s_U)
logging.info(f'Time={time_final-time_initial} to calculate {repeat} {N}*{N} loop Hafnian squared of interest for fixed n random U')




# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
lhaf2s_n[::-1].sort()
lhaf2s_U[::-1].sort()
haf2s_U[::-1].sort()

if plotting:
    plt.figure('lhaf squared')

    plt.plot(list(range(repeat)), lhaf2s_n, '-', label='|lhaf|^2 over n')
    plt.plot(list(range(repeat)), lhaf2s_U, '--', label='|lhaf|^2 over U')
    plt.plot(list(range(repeat)), haf2s_U, '--', label='|haf|^2 over U')

    plt.xticks([0, repeat])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Instances')
    plt.ylabel('|lHaf|^2')
    plt.title('Distribution of unnormalised |lHaf|^2 over fixed U, random n vs fixed n, random U')

    plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_lhaf2.png'))

    plt.figure('normalised lhaf squared')

    lhaf2s_n = lhaf2s_n / np.sum(lhaf2s_n)
    lhaf2s_U = lhaf2s_U / np.sum(lhaf2s_U)
    haf2s_U = haf2s_U / np.sum(haf2s_U)


    plt.plot(list(range(repeat)), lhaf2s_n, '-', label='|lhaf|^2 over n')
    plt.plot(list(range(repeat)), lhaf2s_U, '--', label='|lhaf|^2 over U')
    plt.plot(list(range(repeat)), haf2s_U, '--', label='|haf|^2 over U')

    plt.xticks([0, repeat])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Instances')
    plt.ylabel('|lHaf|^2')
    plt.title('Distribution of normalised |lHaf|^2 over fixed U, random n vs fixed n, random U')

    plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_lhaf2_normalised.png'))