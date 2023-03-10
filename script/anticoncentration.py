import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt
import numba

from src.utils import LogUtils, DFUtils
from thewalrus import perm, hafnian


def Gaussian_lhafs(w, N, var, repeats, func='lhaf', print_bool=False):
    """
    Calculates |lhaf(XX^T, w sumX)| or |haf(XX^T)| or |Per(X)| or |Det(X)| for N*N complex Gaussian matrix X from G(0,var)
    :param w: diagonal weight
    :param N: Dimension of matrix
    :param var: Variance of Gaussian matrix
    :param repeats: Number of repetitions
    :param func: 'lhaf' or 'haf' or 'perm' or 'det'
    :param print_bool: whether print time to console

    :return: array of length `repeats'
    """

    raw = np.zeros(repeats, dtype=float)
    for i in range(repeats):
        X = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N, N)) + \
            1j * np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N, N))

        t0 = time.time()
        if func == 'lhaf':
            loop = (w != 0)
            B = X @ X.T
            gamma = w * np.sum(X, axis=1)

            raw_i = np.absolute(hafnian(B + (gamma - B.diagonal()) * np.eye(N), loop=loop))

        elif func == 'haf':
            B = X @ X.T
            raw_i = np.absolute(hafnian(B, loop=False))

        elif func == 'perm':
            raw_i = np.absolute(perm(X))

        elif func == 'det':
            raw_i = np.absolute(np.linalg.det(X))

        else:
            raise ValueError('Func not recognized')

        raw[i] = raw_i
        t1 = time.time()
        if print_bool:
            print(f'for {i}-th repeat, {func} = {raw_i} in time={t1 - t0}')

    return raw


# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
# func = 'lhaf'
# if func == 'lhaf':
#     w = 1
# else:
#     w = 0
var = 1  # For now we only care about Gaussian matrices with variance 1. This will involve some rescaling of the matrices.
Ns = np.arange(6, 30, step=2)
total_repeats = 100000  # Take some integer multiple of 1000
print_bool = False

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

LogUtils.log_config(time_stamp=time_stamp, filehead='log', module_name='', level=logging.INFO)
logging.info(
    f'Benchmark Anticoncentration for different functions over complex Gaussian matrices of mean 0 and variance {var}.'
    f'Loop Hafnian (and Hafnian) is calculated as |lhaf(XX^T, w(sumX))| for different w (w=0). '
    f'Permanent and determinant is calculated as |Perm(X)| or |Det(X)|. For each dimension from {Ns}, {total_repeats}'
    f'random X-s are generated to calculate the function values. The values are then stored as np arrays in different '
    f'directories marked by timestamp {time_stamp}')


# <<<<<<<<<<<<<<<<<<< Calculating  >>>>>>>>>>>>>>>>>>
# Unable to parallelize this
# @numba.jit(parallel=True)
def wrapper_parallel(N, sub_repeats=1000, func='lhaf', w=0, w_string='0'):
    if total_repeats % sub_repeats != 0:
        raise ValueError('Please make my life easier by making total repeats an integer multiple of sub_repeats')

    if func == 'lhaf':
        if w == 0:
            raise ValueError('You want haf')
        elif w == 1:
            w_string = '1'
        elif w_string=='0':
            raise ValueError('Give a valid w_string')

        save_dir = fr'..\Results\{func}_w={w_string}_{time_stamp}\N={N}'
    else:
        save_dir = fr'..\Results\{func}_{time_stamp}\N={N}'

    n = total_repeats // sub_repeats
    # for i in numba.prange(n):
    for i in range(n):
        t_i = time.time()
        raw = Gaussian_lhafs(w, N, var, sub_repeats, func, print_bool)
        np.save(DFUtils.create_filename(save_dir + rf'\raw_{i}.npy'), raw)
        t_f = time.time()
        logging.info(f'Calculate {i}-th batch {sub_repeats} {func}s for N={N}, w={w} took time={t_f - t_i}')


for iter, N in enumerate(Ns):
    wrapper_parallel(N, sub_repeats=1000, func='det')
    wrapper_parallel(N, sub_repeats=1000, func='lhaf', w=1)
    wrapper_parallel(N, sub_repeats=1000, func='haf')
    wrapper_parallel(N, sub_repeats=1000, func='perm')
    wrapper_parallel(N, sub_repeats=1000, func='lhaf', w=1/N, w_string='1/N')
