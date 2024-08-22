import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time
import matplotlib.pyplot as plt
import numba
import os

from src.utils import LogUtils, DFUtils
from thewalrus import perm, hafnian


def Gaussian_lhafs(w, N, var, repeats, func='lhaf', print_bool=False):
    """
    Calculates |lhaf(XX^T, w sumX)| or |haf(XX^T)| or |Per(X)| for N*N complex Gaussian matrix X from G(0,var)
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

def Gaussian_prods(N, var, repeats):

    X = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(repeats, N))
    Y = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(repeats, N))

    raw = np.prod(np.square(X) + np.square(Y), axis=1)

    return raw


@numba.njit(parallel=True)
def Gaussian_dets(N, var, repeats, func='det', w=1.):
    """
    Calculates |Det(X)| for N*N complex Gaussian matrix X from G(0,var). Optional diagonal weight w.
    This can be parallelized.
    :param N: Dimension of matrix
    :param var: Variance of Gaussian matrix
    :param repeats: Number of repetitions
    :param func: 'det'
    :param w: diagonal weight

    :return: array of length `repeats'
    """

    raw = np.zeros(repeats)
    for i in numba.prange(repeats):
        X = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N, N)) + \
            1j * np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N, N))

        np.fill_diagonal(X, w * np.diagonal(X))

        if func == 'det':
            raw_i = np.absolute(np.linalg.det(X))
        else:
            raise ValueError('Func not recognized')
        raw[i] = raw_i

    return raw

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
# func = 'lhaf'
# if func == 'lhaf':
#     w = 1
# else:
#     w = 0
var = 1  # For now we only care about Gaussian matrices with variance 1. This will involve some rescaling of the matrices.
print_bool = False

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
LogUtils.log_config(time_stamp=time_stamp, filehead='anticoncentration_over_X', module_name='', level=logging.INFO)
# logging.info(
#     f'Benchmark Anticoncentration for different functions over complex Gaussian matrices of mean 0 and variance {var}.'
#     f'Loop Hafnian (and Hafnian) is calculated as |lhaf(XX^T, w(sumX))| for different w (w=0). '
#     f'Permanent and determinant is calculated as |Perm(X)| or |Det(X)|. The values are then stored as np arrays in '
#     f'different directories marked by timestamp {time_stamp}')

logging.info(f'Calculate prod(x_i^2+ y_i^2) for i=1 to N for number of repeats. The values are then stored as np arrays.')

@numba.njit(parallel=True)
def wrapper_prods(N, repeats):

    raw = np.zeros(repeats)
    for i in numba.prange(repeats):
        X = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N))
        Y = np.random.normal(loc=0, scale=np.sqrt(var) / np.sqrt(2), size=(N))

        raw[i] = np.prod(np.square(X) + np.square(Y))

    return raw

# # <<<<<<<<<<<<<<<<<<< Calculating  >>>>>>>>>>>>>>>>>>
# # Unable to parallelize this for loop Hafnian and permanent
# def wrapper(N, total_repeats, sub_repeats, func='lhaf', w=0., w_string='0'):
#     dir_head = fr'..\Results\anticoncentration_over_X\{total_repeats}repeats'
#
#     if total_repeats % sub_repeats != 0:
#         raise ValueError('Please make my life easier by making total repeats an integer multiple of sub_repeats')
#
#     if func == 'lhaf':
#         if w == 0.:
#             raise ValueError('You want haf')
#         elif w_string=='0':
#             raise ValueError('Give a valid w_string')
#
#         save_dir = dir_head + fr'\{func}_w={w_string}_{time_stamp}\N={N}'
#     else:
#         if w == 0:
#             raise Warning('You are setting a zero-diagonal')
#
#         save_dir = dir_head + fr'\{func}_w={w_string}_{time_stamp}\N={N}'
#
#     n = total_repeats // sub_repeats
#
#     for i in range(n):
#         t_i = time.time()
#         if func == 'det':
#             raw = Gaussian_dets(N, var, sub_repeats, func=func, w=w)
#         else:
#             raw = Gaussian_lhafs(w, N, var, sub_repeats, func, print_bool)
#         np.save(DFUtils.create_filename(save_dir + rf'\raw_{i}.npy'), raw)
#         t_f = time.time()
#         logging.info(f'Calculate {i}-th batch {sub_repeats} {func}s for N={N}, w={float(w):.3} took time={t_f - t_i}')
#

# for N in np.arange(6, 30, step=2):
#     wrapper(N, total_repeats=100000, sub_repeats=1000, func='lhaf', w=N**0.25, w_string='N^0.25')
# for N in np.arange(6, 30, step=2):
#     wrapper(N, total_repeats=100000, sub_repeats=1000, func='lhaf', w=0.1*N**0.25, w_string='0.1N^0.25')
# for N in np.arange(30, 40, step=1):
#     wrapper(N, total_repeats=1, sub_repeats=1, func='lhaf', w=1, w_string='1')
# for N in np.arange(6, 29):
#     wrapper(N, total_repeats=100000, sub_repeats=1000, func='lhaf', w=0.4, w_string='0.4')
# for N in Ns:
#     wrapper(N, sub_repeats=1000, func='det')
# for N in Ns:
#     wrapper(N, sub_repeats=1000, func='lhaf', w=0.1, w_string='0.1')

total_repeats = 100000
sub_repeats = 100000
n = total_repeats // sub_repeats
dir_head = fr'..\Results\anticoncentration_over_X\{total_repeats}repeats'
for N in np.logspace(1, 6, num=10, base=10, dtype=int):
    save_dir = dir_head + fr'\prod_{time_stamp}\N={N}'
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    for i in range(n):
        raw = wrapper_prods(N, total_repeats)

        np.save(save_dir + rf'\raw_{i}.npy', raw)

    logging.info(f'Calculate {total_repeats} repeats for N={N} took {time.time() - t0}s')