import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import LogUtils

# This script sets up an experiment and calculates the probabilities
# TODO: make the np arrays shorter before saving, and try to maybe parallelize this.
# TODO: for each (M,N) run a bunch of experiments with different unitaries and try averaging.
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\experiments\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Generate experiment and collect collisionless probabilities with displacement. '
             'The probabilities are unnormalised and only give |lhaf|^2. '
             'The experiment is repeated many times for different unitaries.')

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
M = 16
sq_dis_ratio = 1
# save_fig = True

N = np.sqrt(M)  # Mean photon number
N_int = int(2 * np.floor(N / 2))  # Find the nearest smaller even number otherwise hafnian gives 0
num_prob = math.comb(M, N_int)
logging.info('')
logging.info('For M={}, mean photon N={}, N of interest = {}, sq/dis ratio = {}'.format(M, N, N_int, sq_dis_ratio))

# <<<<<<<<<<<<<<<<<<< Generate experiment  >>>>>>>>>>>>>>>>>>
beta = np.sqrt(N / (M * (sq_dis_ratio + 1)))
r = np.arcsinh(np.sqrt(sq_dis_ratio * beta ** 2))
betas = beta * np.ones(M)
rs = r * np.ones(M)
logging.info('Identical sq = {} and dis = {} in each mode in sdu model'.format(r, beta))

repeat = 100

for unitary_i in range(repeat):
    logging.info('')
    logging.info('{}-th repetition'.format(unitary_i))

    U = unitary_group.rvs(M)  # This method returns random unitary matrix sampled from Haar measure
    np.save(dir + r'\U_{}.npy'.format(unitary_i), U)

    displaced_gbs = sduGBS(M)
    displaced_gbs.add_all(rs, betas, U)

    # gbs = sduGBS(M)  # No displacement
    # gbs.add_all(rs, np.zeros_like(betas), U)

    displaced_p0 = displaced_gbs.vacuum_prob()
    # p0 = gbs.vacuum_prob()
    logging.info(f'Vacuum probability with displacement = {displaced_p0}')
    # <<<<<<<<<<<<<<<<<<< Calculate bunch of probabilities  >>>>>>>>>>>>>>>>>>
    base_outcome = [1] * N_int + [0] * (M - N_int)
    displaced_probs = np.zeros(num_prob)  # unnormalised |lhaf|^2
    # probs = np.zeros(num_prob)  # unnormalised |haf|^2
    time_init = time.time()
    for i, outcome in enumerate(multiset_permutations(base_outcome)):
        time1 = time.time()
        lhaf2 = np.absolute(displaced_gbs.lhaf(outcome, displacement=True)) ** 2
        # haf2 = np.absolute(displaced_gbs.lhaf(outcome, displacement=False))**2
        time2 = time.time()
        displaced_probs[i] = lhaf2
        # probs[i] = haf2
        print(f'Time = {time2 - time1}, lhaf**2 = {lhaf2:.3}')
        # print(f'Time = {time2-time1}, lhaf**2 = {lhaf2:.3}, haf**2 = {haf2:.3}')
    time_final = time.time()

    # prob_int = np.sum(probs)
    # theory_sum = math.comb(M//2 + N_int//2 - 1, N_int//2) * np.tanh(r)**N_int
    # print('theory sum is {}'.format(np.isclose(prob_int, theory_sum)))
    # logging.info(f'Time to calculate {num_prob} probs is {time_final-time_init}. Total prob of collisionless N-photon '
    #              f'coincidence is {prob_int:.3}')
    logging.info(f'Time to calculate {num_prob} probs is {time_final - time_init}.')
    # logging.info(f'Sum of unnormalised undisplaced |haf|^2 {prob_int:.3}, theory (M/2+N/2-1 choose N/2) * tanh(r)^N = {theory_sum:.3}')

    # <<<<<<<<<<<<<<<<<<< Save  >>>>>>>>>>>>>>>>>>
    np.save(dir + fr'\dis_M={M}_N={N_int}_U_{unitary_i}.npy', displaced_probs)
    # np.save(dir+fr'\undis_M={N}_N={N_int}_U_{unitary_i}.npy', probs)

    # <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
    dis_total_prob = sum(displaced_probs)
    dis_probs_norm = displaced_probs / dis_total_prob  # normalise
    dis_probs_norm[::-1].sort()

    # undis_total_prob = sum(probs)
    # undis_probs_norm = probs / undis_total_prob  # normalise
    # undis_probs_norm[::-1].sort()

    # plt.figure(f'normalised |lhaf|^2 for M={M}, N={N_int}')
    # hist, bins = np.histogram(dis_probs_norm, bins=100)
    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    # plt.hist(dis_probs_norm, weights=np.ones(num_prob) / num_prob, alpha=0.5, bins=logbins,
    #          label=f'{unitary_i}')
    # plt.xscale('log')
    # plt.xlabel('Normalised |lhaf|^2')
    # plt.xlim(left=1e-10, right=1e0)
    # plt.title('Displaced M={},N={}'.format(M, N_int))
    # plt.legend()
    # # if save_fig:
    # #     plt.savefig(dir + r'\plots\hist_M={}_N={}.pdf'.format(M, N))
    #
    #
    # plt.figure(f'normalised |haf|^2 for M={M}, N={N_int}')
    # hist2, bins2 = np.histogram(undis_probs_norm, bins=100)
    # logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
    # plt.hist(undis_probs_norm, weights=np.ones(num_prob) / num_prob, alpha=0.5, bins=logbins2,
    #          label=f'{unitary_i}')
    # plt.xscale('log')
    # plt.xlabel('Normalised |haf|^2')
    # plt.xlim(left=1e-10, right=1e0)
    # plt.title('Undisplaced M={},N={}'.format(M, N_int))
    # plt.legend()
    # # if save_fig:
    # #     plt.savefig(dir + r'\plots\hist_M={}_N={}.pdf'.format(M, N))

    plt.figure(f'|lhaf|^2 vs |haf|^2 for M={M}, N={N_int}')
    plt.plot(list(range(num_prob)), dis_probs_norm, ',')
    # plt.plot(list(range(num_prob)), undis_probs_norm, linestyle='dotted', label=f'Undisplaced{unitary_i}')
    plt.yscale('log')
    plt.title('M={}, N={}'.format(M, N_int))
    plt.xticks([0, num_prob])
    # plt.legend()
    # if save_fig:
    #     plt.savefig(dir + r'\plots\lhaf_vs_haf_M={}_N={}.pdf'.format(M, N))  #

plt.axhline(y=1 / num_prob, xmin=0, xmax=num_prob, color='black', label=f'1/({M} choose {N_int})')
plt.axhline(y=0.01 / num_prob, xmin=0, xmax=num_prob, color='red', label=f'0.01/({M} choose {N_int})')
