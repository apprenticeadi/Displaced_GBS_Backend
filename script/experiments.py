import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import LogUtils

# This script sets up an experiment and calculates the probabilities
# TODO: make the np arrays shorter before saving, and try to maybe parallelize this.
# TODO: for each (M,N) run a bunch of experiments with different unitaries and try averaging.
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\experiments\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Generate bunch of experiments and collect collisionless probabilities with and without displacement. '
             'The probabilities are unnormalised and only give |lhaf|^2 and |haf|^2')

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
Ms = [20]
sq_dis_ratio = 1
logging.info('Modes = {}, sq/dis ratio = {}'.format(Ms, sq_dis_ratio))

for M in Ms:
    N = M**0.1  # Mean photon number
    N_int = 2 * int(N/2 + 0.5)  # Find the nearest even number otherwise hafnian gives 0
    logging.info('')
    logging.info('For M={}, mean photon N={}, N of interest = {}'.format(M, N, N_int))

    # <<<<<<<<<<<<<<<<<<< Generate experiment  >>>>>>>>>>>>>>>>>>
    beta = np.sqrt(N/(M*(sq_dis_ratio+1)))
    r = np.arcsinh(np.sqrt(sq_dis_ratio * beta**2))
    betas = beta * np.ones(M)
    rs = r * np.ones(M)
    logging.info('Identical sq = {} and dis = {} in each mode in sdu model'.format(r, beta))

    U = unitary_group.rvs(M)  # This method returns random unitary matrix sampled from Haar measure

    displaced_gbs = sduGBS(M)
    displaced_gbs.add_all(rs, betas, U)

    gbs = sduGBS(M)  # No displacement
    gbs.add_all(rs, np.zeros_like(betas), U)

    displaced_p0 = displaced_gbs.vacuum_prob()
    p0 = gbs.vacuum_prob()
    logging.info(f'Vacuum probability with displacement = {displaced_p0}, without = {p0}')
    # <<<<<<<<<<<<<<<<<<< Calculate bunch of probabilities  >>>>>>>>>>>>>>>>>>
    base_outcome = [1] * N_int + [0] * (M - N_int)
    num_prob = math.comb(M, N_int)
    displaced_probs = np.zeros(num_prob)  # unnormalised |lhaf|^2
    probs = np.zeros(num_prob)  # unnormalised |haf|^2
    time_init = time.time()
    for i, outcome in enumerate(multiset_permutations(base_outcome)):
        time1 = time.time()
        lhaf2 = np.absolute(displaced_gbs.lhaf(outcome, displacement=True))**2
        haf2 = np.absolute(displaced_gbs.lhaf(outcome, displacement=False))**2
        time2 = time.time()
        displaced_probs[i] = lhaf2
        probs[i] = haf2
        print(f'Time = {time2-time1}, lhaf**2 = {lhaf2:.3}, haf**2 = {haf2:.3}')
    time_final = time.time()

    prob_int = np.sum(probs)
    theory_sum = math.comb(M//2 + N_int//2 - 1, N_int//2) * np.tanh(r)**N_int
    print('theory sum is {}'.format(np.isclose(prob_int, theory_sum)))
    # logging.info(f'Time to calculate {num_prob} probs is {time_final-time_init}. Total prob of collisionless N-photon '
    #              f'coincidence is {prob_int:.3}')
    logging.info(f'Time to calculate {num_prob} probs (twice, with and without displacement) is {time_final - time_init}.')
    logging.info(f'Sum of unnormalised undisplaced |haf|^2 {prob_int:.3}, theory (M/2+N/2-1 choose N/2) * tanh(r)^N = {theory_sum:.3}')
    # <<<<<<<<<<<<<<<<<<< Save  >>>>>>>>>>>>>>>>>>
    np.save(dir+r'\dis_M={}_N={}.npy'.format(M, N_int), displaced_probs)
    np.save(dir+r'\undis_M={}_N={}.npy'.format(M, N_int), probs)








