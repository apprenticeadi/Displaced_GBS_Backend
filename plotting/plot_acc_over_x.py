import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
import os

from src.utils import DFUtils
from src.photon_number_distributions import big_F


def prefactor(N, func):
    if func == 'det' or func == 'perm':
        return 1 / np.sqrt(factorial(N))
    elif func == 'haf':
        if N % 2 ==0 :
            return 1 / np.sqrt(comb(N-1, N/2) * factorial(N))
        else:
            return 0

    elif func[:4] == 'lhaf':
        if func[func.index('w')+2:] == "N^-1":
            w = 1 / N
        else:
            w = float(func[func.index('w')+2:])
        return np.sqrt( np.power(2., float(N)) / factorial(N) / big_F(w, N, N)[N])

    else:
        raise ValueError('func not recognized')

#
#
# funcs = ['det', 'perm', 'haf', 'lhaf_w=1', 'lhaf_w=N^-1']
# func_labels={
#     'det': r'Det',
#     'perm': r'Per',
#     'haf': r'Haf',
#     'lhaf_w=1': r'lHaf$(w=1)$',
#     'lhaf_w=N^-1': r'lHaf$(w=\frac{1}{N})$'
# }
#
# repeat_i = 1000  #number of data in each raw data file
# repeats=100000
# N = 20
#
# dir_head = fr'..\Results\anticoncentration_over_X\{repeats}repeats'
# func_dirs = os.listdir(dir_head)
#
# plt.figure(f'Refactorised data N={N}')
# raw_data_dict = {}
# data_dict = {}
# for func in funcs:
#     filtered_dir = [file_ for file_ in func_dirs if file_.startswith(func)][0]  # For now just take the first directory, but later need to modify this when data become more complicated
#     N_dir = dir_head + rf'\{filtered_dir}\N={N}'
#
#     raw_data_files = os.listdir(N_dir)
#
#     combined_raw_data = np.zeros((len(raw_data_files), repeat_i), dtype=float)
#
#     for i, fn in enumerate(raw_data_files):
#         combined_raw_data[i, :] = np.load(N_dir + fr'\{fn}')
#
#     combined_raw_data = combined_raw_data.flatten()
#     combined_raw_data.sort()
#     raw_data_dict[func] = combined_raw_data
#
#     pf = prefactor(N, func)
#     refactored_data = pf * combined_raw_data
#     data_dict[func] = refactored_data
#
#     plt.plot(list(range(repeats)), refactored_data, label=func_labels[func])
#
# plt.xticks([1, repeats])
# plt.legend()
# plt.yscale('log')
#
#
# # # Here I refactorise the data such that their second moment is 1. Recipe known for permanent and det but not for lhaf and haf
# # renorm_data_dict = {}
# # plt.figure(f'Renormalised data N={N}')
# # for func in raw_data_dict.keys():
# #     second_moment = np.mean(raw_data_dict[func]**2)
# #     renorm_data = raw_data_dict[func] / np.sqrt(second_moment)
# #     renorm_data_dict[func] = renorm_data
# #     plt.plot(list(range(repeats)), renorm_data, label=func)
# #
# # plt.xticks([1, repeats])
# # plt.legend()
# # plt.yscale('log')
#
# # squared data and probability density function
# x_min = np.min(data_dict['lhaf_w=1'])**2
# x_min_log = np.floor(np.log10(x_min))  # the minimum value much more likely to come from lhaf with w=1
# xs = np.logspace(start=x_min_log, stop=1, num=16*int(np.absolute(x_min_log)))  # evenly distributed in log scale, 16 per decade
#
# prob_density_function = {}  # for the squared data
# prob_density_function['xs'] = xs[1:-1]  # f(x_i) = (F(x_i+1) - F(x_i-1) ) / (x_i+1 - x_i-1)
# prob_distrib_function = {}
# prob_distrib_function['xs'] = xs
# squared_data = {}
#
# for func in data_dict.keys():  # use refactorised not renormalised data
#     func_squared_data = data_dict[func] ** 2
#     squared_data[func] = func_squared_data  # This is already sorted
#
#     func_distrib = np.zeros_like(xs, dtype=float)
#     for i, x in enumerate(xs):
#         func_distrib[i] = np.argmax(func_squared_data > x) / len(func_squared_data)
#     prob_distrib_function[func] = func_distrib
#
#     func_pdf = np.zeros(len(xs) - 2, dtype=float)
#     for i,x in enumerate(xs[1:-1]):
#         func_pdf[i] = (func_distrib[i+1] - func_distrib[i-1]) / (xs[i+1] - xs[i-1])
#     prob_density_function[func] = func_pdf
#
#     plt.figure(f'distribution N={N}')
#     plt.plot(xs, func_distrib, '.', label=func_labels[func])
#
#     plt.figure(f'pdf N={N}')
#     arg_nonzero = np.where(func_pdf > 0)
#     plt.plot(xs[1:-1][arg_nonzero], func_pdf[arg_nonzero], label=func_labels[func])
#
# plt.figure(f'distribution N={N}')
# plt.title('Cumulative distribution functions')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel(r'$x$')
# plt.ylabel(r'$F(x)$')
# plt.legend()
#
# plt.figure(f'pdf N={N}')
# plt.title('Probability density function')
# plt.yscale('log')
# plt.xscale('linear')
# plt.xlim(-0.1, 2)
# plt.legend()