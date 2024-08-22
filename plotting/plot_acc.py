import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from scipy.stats import bootstrap
import os
import datetime
import time
import pandas as pd

from src.utils import DFUtils, DGBSUtils
from src.photon_number_distributions import big_F

plt.show()
plt.ion()
matplotlib.use('TkAgg')

# <<<<<<<<<<<<<<<<<<< Functions  >>>>>>>>>>>>>>>>>>
# def read_refactored_acc_files(funcs, repeats_list):
#     multi_func_raw_data = read_raw_acc_files(funcs, repeats_list)
#     multi_func_refactored_data = {}
#     for func in multi_func_raw_data.keys():
#         raw_data = multi_func_raw_data[func]
#         multi_func_refactored_data[func] = refactor_data(func, raw_data)
#
#     return multi_func_refactored_data
#
#
# def refactor_data(func, raw_data):
#     refactored_data = {}
#     for N in raw_data.keys():
#         refactored_data[N] = prefactor(N, func) * raw_data[N]
#
#     return refactored_data


def prefactor(_N, _func):
    if _func == 'det' or _func == 'perm':
        return 1 / np.sqrt(factorial(_N))
    elif _func == 'haf':
        if _N % 2 == 0:
            return 1 / np.sqrt(comb(_N - 1, _N / 2) * factorial(_N))
        else:
            return 0

    elif _func[:4] == 'lhaf':
        w_label = _func[5:]
        w = DGBSUtils.read_w_label(w_label, _N)
        return np.sqrt(np.power(2., float(_N)) / factorial(_N) / big_F(w, _N, _N)[_N])
        # return np.sqrt( np.power(2., float(_N)) * comb(_N**2, _N) / big_F(w, _N, _N)[_N] / np.power(float(_N), 2*float(_N)) )

    else:
        raise ValueError(f'{_func} not recognized')


def read_raw_acc_files(funcs, repeats_list):
    multi_func_raw_data = {}
    for func in funcs:
        multi_func_raw_data[func] = multirep_raw_acc_data(func, repeats_list)

    return multi_func_raw_data


def multirep_raw_acc_data(func, repeats_list):
    repeats_list.sort()  # if there are datafiles for the same N in two repeats, we want the higher repeat to overwrite the lower repeat

    multi_rep_data = {}
    for repeats in repeats_list:
        raw_data = raw_acc_data(func, repeats)

        multi_rep_data.update(raw_data)

    return multi_rep_data


def raw_acc_data(func, repeats):
    dir_head = fr'..\Results\anticoncentration_over_X\{repeats}repeats'
    all_func_dirs = os.listdir(dir_head)

    _func_dirs = [file_ for file_ in all_func_dirs if
                  file_.startswith(func + '_')]  # add '_' otherwise w=10 and w=1 get mixed up

    if len(_func_dirs) == 0:
        return {}  # Don't return exception, but empty dictionary. This way, in multi-rep function, we merely update the dictionary with an empty dictionary.

    raw_data = {}
    for _func_dir in _func_dirs:
        for N_str in os.listdir(dir_head + fr'\{_func_dir}'):
            N = int(N_str[2:])

            if N in raw_data.keys():
                continue

            # TODO: fix this. this is a temporary solution for a half-run permanent calculation
            if func == 'perm' and N == 26:
                continue

            N_dir = dir_head + fr'\{_func_dir}\{N_str}'
            raw_data_files = os.listdir(N_dir)

            combined_raw_data = np.zeros((len(raw_data_files), repeats // len(raw_data_files)), dtype=float)
            for i, fn in enumerate(raw_data_files):
                combined_raw_data[i, :] = np.load(N_dir + fr'\{fn}')
            combined_raw_data = combined_raw_data.flatten()
            combined_raw_data.sort()

            raw_data[N] = combined_raw_data

    return raw_data


def get_cdf(data, x, bootstrapping = False, confidence_level=0.9):
    """cumulative distribution function"""

    data_converted = data < x
    cdf = np.mean(data_converted)

    if bootstrapping:
        print('Start bootstrapping')
        t1 = time.time()
        bootstrap_res = bootstrap((data_converted,), np.mean, confidence_level=confidence_level, batch=100, method='basic')
        t2 = time.time()
        # print(f'end bootstrapping after {t2-t1}s')
        n_error = cdf - bootstrap_res.confidence_interval.low
        p_error = bootstrap_res.confidence_interval.high - cdf
    else:
        n_error = 0
        p_error = 0

    return cdf, n_error, p_error


def func_labels(func_string):
    if func_string == 'det':
        # return r'$\frac{|Det(X)|}{\sqrt{N!}}$'
        return 'Det'

    elif func_string == 'perm':
        # return r'$\frac{|Perm(X)|}{\sqrt{N!}}$'
        return 'Perm'

    elif func_string == 'haf':
        # return r'$\frac{|Haf(XX^T)|}{\sqrt{N!\binom{N-1}{N/2}}}$'
        return 'Haf'

    elif func_string[:4] == 'lhaf':

        if func_string[func_string.index('w') + 2:] == 'N^-1':
            return r'lHaf$(w=\frac{1}{N})$'
        elif func_string[func_string.index('w') + 2:] == 'N^0.25':
            return r'lHaf$(w=N^{1/4})$'
        else:
            w_value = func_string[func_string.index('w') + 2:]
            # return fr'$\frac{{lHaf(XX^T, {{{w_value}}}  \sum X)}}{{\sqrt{{N! F_N( {{{w_value}}} ) / 2^N}}  }}$'
            return rf'lHaf$(w={{{w_value}}})$'

    else:
        raise ValueError(f'{func_string} not supported')




''' Basic parameters  '''
repeats_list = [1000, 10000, 100000]
bootstrapping = True
# funcs = ['perm', 'det', 'haf', 'lhaf_w=N^-1', 'lhaf_w=0.1', 'lhaf_w=0.01', 'lhaf_w=1']
funcs =  ['perm', 'det', 'haf', 'lhaf_w=0.1', 'lhaf_w=0.4', 'lhaf_w=1']
# funcs = ['lhaf_w=1', 'lhaf_w=N^0.25']

save_fig = False
time_stamp = r'2024-08-08(10-46-44.108248)'  # datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)") # r'2024-02-26(15-48-23.239095)'  #r'2024-03-07(12-49-07.901770)'
plt_dir = fr'..\Plots\acc_numerics\{time_stamp}'

''' Read data  '''
# This is a dictionary with keys being funcs, and values being a dictionary, whose keys are Ns, and values are np.array
# of absolute function values over corresponding distribution.
# {'det' : {6: [...], 7: [...], ...}, 'perm': {6: [...], 7: [...], ...}, ...}

print('Start to read data')
multi_func_raw_data = read_raw_acc_files(funcs, repeats_list)
# multi_func_refactored_data = read_refactored_acc_files(funcs, repeats_list)

print('Read data finished')
''' Plotting  '''
plt.rcParams.update({'font.size': 8})
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 14
Ns = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28] # , 34] # np.arange(6, 37)
funcs_to_plot = ['det', 'perm', 'haf', 'lhaf_w=0.1', 'lhaf_w=0.4', 'lhaf_w=1']

''' Plot cdf[1/epsilon] against epsilon  '''
epsilons = np.load(r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Plots\acc_numerics\2024-03-07(12-49-07.901770)\data\epsilons.npy') #  np.logspace(start=0, stop=6, num=20, base=10)
# epsilons = np.load(r'..\Plots\acc_numerics\2024-01-08(11-29-21.512801)' + rf'\data\epsilons.npy')
np.save(DFUtils.create_filename(plt_dir + rf'\data\epsilons.npy'), epsilons)

# Set figure configuration according to Ns
fig, axs = plt.subplot_mosaic('aabbcc;ddeeff;ggghhh', figsize=(15, 10), layout='constrained')
# assert len(axs.keys()) - 2 == len(Ns)
#
for i_N, N in enumerate([6, 10, 16, 20, 28, 34]):

    ax = list(axs.values())[i_N]
    ax_label = list(axs.keys())[i_N]
    ax.set_title(rf'({ax_label}) N={N}', fontfamily='serif', loc='left', fontsize=fontsize)

    print(f'Plot cdf(1/epsilon) against epsilon for N={N}')

    for i_func, func in enumerate(funcs_to_plot):
        func_label = func_labels(func)

        if N in multi_func_raw_data[func].keys():
            # # Refactor data
            # raw_data = multi_func_raw_data[func][N]
            # refactored_data = prefactor(N, func) * raw_data
            #
            # cdfs = np.zeros((len(epsilons)), dtype=float)
            # error_bars = np.zeros((2, len(epsilons)), dtype=float)
            #
            # for i_epsilon, epsilon in enumerate(epsilons):
            #
            #     print(f'Calculate cdf for {func} and epsilon={epsilon}')
            #     t1 = time.time()
            #     cdf, n_error, p_error = get_cdf(refactored_data, 1/epsilon, bootstrapping=bootstrapping, confidence_level=0.95)
            #     t2 = time.time()
            #     print(f'Calculate finished after {t2-t1}s. Results={(cdf, n_error, p_error)}')
            #
            #     cdfs[i_epsilon] = cdf
            #     error_bars[:, i_epsilon] = np.array([n_error, p_error])
            #
            # np.save(DFUtils.create_filename(plt_dir + rf'\data\{func}\N={N}_cdfs.npy'), cdfs)
            # np.save(DFUtils.create_filename(plt_dir + rf'\data\{func}\N={N}_errors.npy'), error_bars)

            cdfs = np.load(plt_dir + rf'\data\{func}\N={N}_cdfs.npy')
            error_bars = np.load(plt_dir + rf'\data\{func}\N={N}_errors.npy')

            idx_zero = np.argmax(cdfs==0)
            if idx_zero == 0:
                ax.errorbar(epsilons, cdfs, marker='.', yerr=error_bars, color=cycle[i_func], label=func_label)
            else:
                ax.errorbar(epsilons[:idx_zero], cdfs[:idx_zero], marker='.', yerr=error_bars[:, :idx_zero], color=cycle[i_func], label=func_label)

            # plt.figure(f'cdf for {func}')
            # plt.errorbar(epsilons, cdfs, fmt='x', ls='None', yerr=error_bars, label=f'N={N}')

        else:
            continue

    if i_N >= 3: # last row
        ax.set_xlabel(r'$\alpha$', fontsize=fontsize-2)
    if i_N % 3 == 0: # first column
        ax.set_ylabel(r'$F(N, 1/\alpha)$', fontsize=fontsize-2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-4)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xscale('log')
    ax.set_ylim(0, 2)
    if i_N == 0:
        ax.legend(fontsize=fontsize-1)


''' Bootstrap mean/median/first quartile  '''
def find_quart(a, axis=0):
    return np.quantile(a, 0.25, axis=axis)

ax_list = list(axs.values())[-2:]
ax_key_list = list(axs.keys())[-2:]
etas = [0.5, 0.25]
for label, ax, eta in zip(ax_key_list, ax_list, etas):
    ax.set_title(rf'({label}) $\eta={{{int(eta*100)}}}\%$', fontfamily='serif', loc='left', fontsize=fontsize)

for i_func, func in enumerate(funcs_to_plot):

    # results_rows = []
    # for i_N, N in enumerate(Ns):
    #
    #     if N in multi_func_raw_data[func].keys():
    #         # Refactor data
    #         raw_data = multi_func_raw_data[func][N]
    #         refactored_data = prefactor(N, func) * raw_data
    #
    #         re_mean = np.mean(refactored_data)
    #         re_median = np.median(refactored_data)
    #         re_quart = find_quart(refactored_data)
    #
    #         print(f'Start bootstrapping {func} mean for N={N}.')
    #         t1 = time.time()
    #         mean_boot = bootstrap((refactored_data,), np.mean, confidence_level=0.95, batch=100, method='basic')
    #         t2 = time.time()
    #         print(f'Bootstrapping finished after {t2-t1}s. Start bootstrapping {func} median for N={N}.')
    #         t3 = time.time()
    #         median_boot = bootstrap((refactored_data, ), np.median, confidence_level=0.95, batch=100, method='basic')
    #         t4 = time.time()
    #         print(f'Bootstrapping finished after {t4-t3}s. Start bootstrapping {func} first quart for N={N}.')
    #         t5=time.time()
    #         quart_boot = bootstrap((refactored_data, ), find_quart, confidence_level=0.95, batch=100, method='basic')
    #         t6 = time.time()
    #         print(f'Bootstrapping finished after {t6-t5}s. ')
    #
    #
    #
    #         results_rows.append({'N': N, 'mean': re_mean,
    #                              'mean_n_error': re_mean - mean_boot.confidence_interval.low,
    #                              'mean_p_error': mean_boot.confidence_interval.high - re_mean,
    #                              'median': re_median,
    #                              'median_n_error': re_median - median_boot.confidence_interval.low,
    #                              'median_p_error': median_boot.confidence_interval.high - re_median,
    #                              'quart': re_quart,
    #                              'quart_n_error': re_quart - quart_boot.confidence_interval.low,
    #                              'quart_p_error': quart_boot.confidence_interval.high - re_quart,
    #                              }
    #                             )
    #     else:
    #         continue

    # results_df = pd.DataFrame(results_rows)
    # results_df.to_csv(DFUtils.create_filename(plt_dir + rf'\data\{func}\bootstrapping_df.csv'), index=False)

    results_df = pd.read_csv(plt_dir + rf'\data\{func}\bootstrapping_df.csv')

    #
    # plt.figure('Mean')
    # plt.errorbar(results_df['N'], results_df['mean'], yerr=[results_df['mean_n_error'],
    #                                                         results_df['mean_p_error']],
    #              capsize=5, label=func_labels(func))

    for i_q, quantity in enumerate(['median', 'quart']):
        results = np.array(results_df[quantity])
        n_errors = np.array(results_df[f'{quantity}_n_error'])
        p_errors = np.array(results_df[f'{quantity}_p_error'])

        ax = ax_list[i_q]
        if func == 'lhaf_w=0.1':
            marker = 'x'
            ls = 'None'
        else:
            marker = '.'
            ls = '-'
        ax.errorbar(results_df['N'], 1/results, yerr = [n_errors / results**2, p_errors / results**2],
                    color=cycle[i_func],
                    marker=marker, ls=ls, label=func_labels(func) )

        ax.set_xlabel('$N$', fontsize=fontsize-2)
        if i_q == 0:
            ax.set_ylabel(r'$\alpha$', fontsize=fontsize-2)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
        ax.set_yscale('log')
        ax.set_ylim([1, 1e4])

#
# fig2=plt.figure('Mean')
# plt.legend(fontsize=fontsize-4, loc='upper right')
# plt.ylim([0, 1])
# plt.xlabel('$N$', fontsize=fontsize)
# plt.ylabel('Refactored mean', fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=fontsize-4)
# fig2.set_size_inches(10, 6, forward=True)
# if save_fig:
#     plt.savefig(plt_dir + rf'\means.pdf')

if save_fig:
    fig.savefig(plt_dir + rf'\combined_plot.pdf')

# for func in funcs_to_plot:
#     plt.figure(f'cdf for {func}')
#     plt.xlabel(r'$\epsilon$', fontsize=fontsize)
#     plt.ylabel(r'$F(N, 1/\epsilon)$', fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=fontsize-4)
#     # plt.xticks([0, 1e6])
#     plt.yscale('linear')
#     plt.xscale('linear')
#     plt.ylim(-0.1, 1.1)
#     plt.xlim(0, 100)
#     if func == 'det':
#         plt.legend(fontsize=fontsize-6)
#     if save_fig:
#         plt.savefig(DFUtils.create_filename(plt_dir + rf'\cdf_for_{func}.pdf'))


''' Plot 1-cdf against N  '''
# special_eps_id = [0, 3, 6]
#
# for eps_id in special_eps_id:
#     epsilon = epsilons[eps_id]
#
#     plt.figure(rf'1-cdf for eps={epsilon}')
#
#     for func in funcs_to_plot:
#         inv_cdfs = []
#         for N in Ns:
#             if N in multi_func_raw_data[func].keys():
#                 cdfs = np.load(plt_dir + rf'\data\{func}\N={N}_cdfs.npy')
#                 error_bars = np.load(plt_dir + rf'\data\{func}\N={N}_errors.npy')
#
#                 inv_cdfs.append({
#                     'N': N,
#                     '1-cdf': 1 - cdfs[eps_id],
#                     'n_error': error_bars[1, eps_id],
#                     'p_error': error_bars[0, eps_id]
#                 })
#
#             else:
#                 continue
#
#         inv_cdfs_df = pd.DataFrame(inv_cdfs)
#         inv_cdfs_df.to_csv(DFUtils.create_filename(plt_dir + rf'\data\{func}\1-cdf(N, eps={epsilon}).csv'))
#
#         plt.errorbar(inv_cdfs_df['N'], np.log(list(inv_cdfs_df['1-cdf'])), yerr=np.asarray([inv_cdfs_df['n_error'], inv_cdfs_df['p_error']]) / np.asarray(inv_cdfs_df['1-cdf']),
#                      capsize=1.5, label=func_labels(func), ls='None', fmt='x')
#
#     plt.legend(fontsize=fontsize - 4, loc='lower left')
#     plt.ylabel(rf'$\ln(1-F(N, {{{1/epsilon:.2f}}})$', fontsize=fontsize)
#     plt.xlabel('$N$', fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=fontsize - 4)
#     plt.savefig(plt_dir + rf'\1-cdf(N, eps={epsilon:.2f}).pdf')
#     plt.ylim(-6, 0.5)

