import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt

from src.ACC_functions import lhaf_squared
from src.utils import LogUtils, DFUtils

# This script sets up an experiment and calculates the probabilities
# TODO: parallelize this
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\anticoncentration\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Benchmark Anticoncentration for two weightings, w1 = 1/sqrt(M), and w2=1. '
             'Raw data files are saved for each (M,N). Each row is a unitary U drawn from Haar measure. '
             'Each row consists of 3 values: [lhaf^2 for w1, lhaf^2 for w2, haf^2]. haf^2 doesnt care about w value')

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
Ns = np.arange(4, 33, step=2)

repeat = 1000

logging.info(f'N takes {Ns}, repeat = {repeat}')

save_U = False
plotting = False
if plotting:
    plot_dir = dir + r'\plots'


for iter, N in enumerate(Ns):

    M = N ** 2  # modes

    # consider two different diagonal weights
    w1 = 1/np.sqrt(M)
    w2 = 1.

    logging.info('')
    logging.info(f'For M={M}, N={N}, w1={w1} and w2={w2}')

    dir_raw = dir + fr'\M={M}_N={N}'

    # Each row is [lhaf^2(w1), lhaf^2(w2), haf^2(w1)]  Note hafnian^2 doesn't care about w value
    lhaf2s = np.zeros((repeat, 3), dtype=float)
    # <<<<<<<<<<<<<<<<<<< Calculate bunch of unitaries  >>>>>>>>>>>>>>>>>>
    time_intial = time.time()
    for unitary_i in range(repeat):

        U = unitary_group.rvs(M)  # This method returns random unitary matrix sampled from Haar measure
        if save_U:
            np.save( DFUtils.create_filename(dir_raw + rf'\U_{unitary_i}.npy'), U)

        t1 = time.time()
        lhaf2_1 = lhaf_squared(U, w1, N, loop=True)
        lhaf2_2 = lhaf_squared(U, w2, N, loop=True)
        haf2 = lhaf_squared(U, w2, N, loop=False)
        t2 = time.time()

        lhaf2s[unitary_i] = np.asarray([lhaf2_1, lhaf2_2, haf2])

        logging.info(
            f'{unitary_i}-th rep, lhaf^2(w1)={lhaf2_1:.3}, lhaf^2(w2)={lhaf2_2:.3}, haf^2={haf2:.3}, time is {t2 - t1:.3}')

    time_final = time.time()
    logging.info(f'N={N}, time to calculate {repeat}*3 lhaf^2 is {time_final - time_intial:.3}')

    np.save(dir_raw + r'_lhaf2_raw.npy', lhaf2s)

    if plotting:

        # Sort the arrays
        for col in range(lhaf2s.shape[1]):
            lhaf2s[:, col][::-1].sort()

        repeats = list(range(repeat))
        plt.figure(f'N={N}_repeat={repeat}_1')
        plt.plot(repeats, lhaf2s[:,0], label='lhaf^2(w1)')
        plt.plot(repeats, lhaf2s[:,1], label='lhaf^2(w2)')
        plt.plot(repeats, lhaf2s[:,2], label='haf^2')

        plt.yscale('log')
        plt.title(f'M={M},N={N},w1={w1:.2},w2={w2:.2}')
        plt.xticks([0, repeat])
        plt.legend()
        plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_M={M}_N={N}.png'))



