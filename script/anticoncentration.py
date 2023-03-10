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

    :return: array of lenght `repeats'
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
func = 'lhaf'
var = 1  # For now we only care about Gaussian matrices with variance 1. This will involve some rescaling of the matrices.
if func == 'lhaf':
    w = 1
else:
    w = 0
Ns = np.arange(6, 32, step=2)
total_repeats = 100000  # Take some integer multiple of 1000
print_bool = False

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
if func == 'lhaf':
    dir = fr'..\Results\anticoncentration_over_X\{func}_w={w}_{time_stamp}'
    LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)
    logging.info(
        f'Benchmark Anticoncentration for the function |lhaf(XX^T, w(sumX))| for w={w}. '
        f'The function is calculated for {total_repeats} random complex Gaussian matrices of mean 0 and variance {var}')
else:
    dir = fr'..\Results\anticoncentration_over_X\{func}_{time_stamp}'
    LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)
    if func == 'haf':
        logging.info(
            f'Benchmark Anticoncentration for the function |haf(XX^T)| '
            f'The function is calculated for {total_repeats} random complex Gaussian matrices of mean 0 and variance {var}')
    else:
        logging.info(
            f'Benchmark Anticoncentration for the function |{func}(X)| '
            f'The function is calculated for {total_repeats} random complex Gaussian matrices of mean 0 and variance {var}')

# <<<<<<<<<<<<<<<<<<< Calculating  >>>>>>>>>>>>>>>>>>
# Unable to parallelize this
# @numba.jit(parallel=True)
def wrapper_parallel(w, N, var, total_repeats, sub_repeats, save_dir, func='lhaf', print_bool=False):
    if total_repeats % sub_repeats !=0:
        raise ValueError('Please make my life easier by making total repeats an integer multiple of sub_repeats')
    n = total_repeats // sub_repeats
    # for i in numba.prange(n):
    for i in range(n):
        t_i = time.time()
        raw = Gaussian_lhafs(w, N, var, sub_repeats, func, print_bool)
        np.save(DFUtils.create_filename(save_dir + rf'\raw_{i}.npy'), raw)
        t_f = time.time()
        logging.info(f'Calculate {i}-th batch {sub_repeats} {func}s for N={N}, w={w}, var={var} took time={t_f - t_i}')

for iter, N in enumerate(Ns):

    save_dir = dir + fr'\raw\N={N}_var={var}'
    wrapper_parallel(w, N, var, total_repeats, 1000, save_dir, func='lhaf', print_bool=False)


