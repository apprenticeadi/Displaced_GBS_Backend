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
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\experiments\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Generate bunch of experiments and collect collisionless probabilities')

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
Ms = list(range(25, 11))
sq_dis_ratio = 1
logging.info('Modes = {}, sq/dis ratio = {}'.format(Ms, sq_dis_ratio))

for M in Ms:
    N = M**0.5  # Mean photon number
    N_int = int(N + 0.5)
    logging.info('')
    logging.info('For M={}, mean photon N={}, N of interest = {}'.format(M, N, N_int))

    # <<<<<<<<<<<<<<<<<<< Generate experiment  >>>>>>>>>>>>>>>>>>
    beta = np.sqrt(N/(M*(sq_dis_ratio+1)))
    r = np.arcsinh(np.sqrt(sq_dis_ratio * beta**2))
    betas = beta * np.ones(M)
    rs = r * np.ones(M)
    logging.info('Identical sq = {} and dis = {} in each mode in sdu model'.format(r, beta))

    U = unitary_group.rvs(M)  # This method returns random unitary matrix sampled from Haar measure

    gbs = sduGBS(M)
    gbs.add_all(rs, betas, U)

    p0 = gbs.vacuum_prob()
    logging.info(f'Vacuum probability = {p0}')
    # <<<<<<<<<<<<<<<<<<< Calculate bunch of probabilities  >>>>>>>>>>>>>>>>>>
    base_outcome = [1] * N_int + [0] * (M - N_int)
    num_prob = math.comb(M, N_int)
    probs = np.zeros(num_prob)
    time_init = time.time()
    for i, outcome in enumerate(multiset_permutations(base_outcome)):
        time1 = time.time()
        prob = p0 * gbs.prob(outcome)
        time2 = time.time()
        probs[i] = prob
        print(f'Time = {time2-time1}, prob = {prob:.3}')
    time_final = time.time()
    prob_int = np.sum(prob)
    logging.info(f'Time to calculate {num_prob} probs is {time_final-time_init}. Total prob of collisionless N-photon '
                 f'coincidence is {prob_int:.3}')

    # <<<<<<<<<<<<<<<<<<< Save  >>>>>>>>>>>>>>>>>>
    np.save(dir+r'\M={}_N={}.npy'.format(M, N_int), probs)








