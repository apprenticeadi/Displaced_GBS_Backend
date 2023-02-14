import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import LogUtils, DFUtils, RandomUtils

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
M_min = 9
M_max = 20
Ms = np.arange(M_min, M_max+1)
depth = 4
sq_dis_ratio = 1
alpha = 1  # the alpha in anti-concentration
repeat = 100

# This script sets up an experiment and calculates the probabilities
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\bounded_anticoncentration\M={}-{}_d={}_{}'.format(M_min, M_max, depth, time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Generate bounded degree DisplacedGBS experiment and benchmark Anticoncentration. '
             'The probabilities are unnormalised and only give |lhaf|^2. '
             'The experiment is repeated many times for different unitaries.')

# <<<<<<<<<<<<<<<<<<< Optional parameters  >>>>>>>>>>>>>>>>>>

save_U = True
plotting = True
if plotting:
    plot_dir = dir + r'\plots'

acc_beta_means = np.zeros_like(Ms, dtype=float)
acc_beta_errs = np.zeros_like(Ms, dtype=float)

for iter, M in enumerate(Ms):

    N = np.sqrt(M)  # Mean photon number
    N_int = int(np.floor(N))  # find nearest smaller integer
    num_prob = math.comb(M, N_int)
    logging.info('')
    logging.info(f'For M={M}, depth={depth}, mean photon N={N}, N of interest = {N_int}, sq/dis ratio = {sq_dis_ratio}, num_prob={num_prob}')

    dir2 = dir + fr'\M={M}_N={N_int}_d={depth}'

    # <<<<<<<<<<<<<<<<<<< Generate experiment  >>>>>>>>>>>>>>>>>>
    beta = np.sqrt(N / (M * (sq_dis_ratio + 1)))
    r = np.arcsinh(np.sqrt(sq_dis_ratio * beta ** 2))
    betas = beta * np.ones(M)
    rs = r * np.ones(M)
    logging.info(f'Identical sq = {r} and dis = {beta} in each mode in sdu model')

    all_displaced_probs = np.zeros((repeat, num_prob))
    all_displaced_probs_norm = np.zeros((repeat, num_prob))
    acc_betas = np.zeros(repeat)  # This is the fraction of probabilities above alpha/num_prob
    for unitary_i in range(repeat):

        I = RandomUtils.random_interferometer(M, depth)
        U = I.calculate_transformation()
        if save_U:
            np.save( DFUtils.create_filename(dir2 + rf'\U_{unitary_i}.npy'), U)

        displaced_gbs = sduGBS(M)
        displaced_gbs.add_all(rs, betas, U)

        # displaced_p0 = displaced_gbs.vacuum_prob()
        # logging.info(f'Vacuum probability with displacement = {displaced_p0}')
        # <<<<<<<<<<<<<<<<<<< Calculate bunch of probabilities  >>>>>>>>>>>>>>>>>>
        base_outcome = [1] * N_int + [0] * (M - N_int)
        displaced_probs = np.zeros(num_prob)  # unnormalised |lhaf|^2
        time_init = time.time()
        for i, outcome in enumerate(multiset_permutations(base_outcome)):
            time1 = time.time()
            lhaf2 = np.absolute(displaced_gbs.lhaf(outcome, displacement=True)) ** 2
            time2 = time.time()
            displaced_probs[i] = lhaf2
            print(f'Time = {time2 - time1}, lhaf**2 = {lhaf2:.3}')
        time_final = time.time()

        # Normalise
        dis_total_prob = sum(displaced_probs)
        dis_probs_norm = displaced_probs / dis_total_prob  # normalise
        dis_probs_norm[::-1].sort()

        # Find acc beta
        acc_beta = np.count_nonzero(dis_probs_norm > alpha/num_prob) / num_prob
        acc_betas[unitary_i] = acc_beta

        # <<<<<<<<<<<<<<<<<<< Write into data array  >>>>>>>>>>>>>>>>>>
        all_displaced_probs[unitary_i] = displaced_probs
        all_displaced_probs_norm[unitary_i] = dis_probs_norm
        logging.info(
            f'{unitary_i}-th repetition, time to calculate {num_prob} probs is {time_final - time_init}, sum={dis_total_prob}, acc beta={acc_beta}.')

        # <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
        if plotting:
            plt.figure(f'M={M}, N={N_int}')
            plt.plot(list(range(num_prob)), dis_probs_norm, ',')


    if plotting:
        plt.yscale('log')
        plt.title('M={}, N={}, d={}'.format(M, N_int, depth))
        plt.xticks([0, num_prob])
        plt.axhline(y=1 / num_prob, xmin=0, xmax=num_prob, color='black', label=f'1/({M} choose {N_int})')
        plt.axhline(y= alpha / num_prob, xmin=0, xmax=num_prob, color='red', label=f'0.01/({M} choose {N_int})')
        plt.legend()
        plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_M={M}_N={N_int}.pdf'))

    # <<<<<<<<<<<<<<<<<<< Save data array  >>>>>>>>>>>>>>>>>>
    np.save(DFUtils.create_filename(dir2 + r'\lhaf2.npy'), all_displaced_probs)
    np.save(DFUtils.create_filename(dir2 + r'\norm_probs.npy'), all_displaced_probs_norm)
    np.save(DFUtils.create_filename(dir2 + fr'\M={M}_N={N_int}_accbetas.npy'), acc_betas)

    # <<<<<<<<<<<<<<<<<<< Deal with acc betas  >>>>>>>>>>>>>>>>>>
    acc_beta_mean = np.mean(acc_betas)
    acc_beta_err = np.std(acc_betas)
    acc_beta_means[iter] = acc_beta_mean
    acc_beta_errs[iter] = acc_beta_err
    logging.info(f'For M={M}, N={N}, acc beta mean = {acc_beta_mean}, acc beta std = {acc_beta_err} for {repeat} different unitaries')

np.save(dir + fr'\acc_beta_means.npy', acc_beta_means)
np.save(dir + fr'\acc_beta_errs.npy', acc_beta_errs)

if plotting:
    plt.figure(f'acc_betas')
    plt.errorbar(Ms, acc_beta_means, yerr=acc_beta_errs)
    plt.xlabel('M')
    plt.ylabel('beta')
    plt.ylim([0, 1])
    plt.title('beta against M')
    plt.savefig(plot_dir + fr'\plot_acc_betas.pdf')



