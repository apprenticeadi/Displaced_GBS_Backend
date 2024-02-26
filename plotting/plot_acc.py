import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from scipy.stats import bootstrap
import os
import datetime
import time
import pandas as pd

from src.utils import DFUtils, DGBSUtils
from src.photon_number_distributions import big_F


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
        print(f'end bootstrapping after {t2-t1}s')
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




# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
repeats_list = [10000, 100000]
bootstrapping = False
# funcs = ['perm', 'det', 'haf', 'lhaf_w=N^-1', 'lhaf_w=0.1', 'lhaf_w=0.01', 'lhaf_w=1']
funcs = ['perm', 'det', 'haf', 'lhaf_w=0.1', 'lhaf_w=1']
# funcs = ['lhaf_w=1', 'lhaf_w=N^0.25']

save_fig = True
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plt_dir = fr'..\Plots\acc_numerics\{time_stamp}'

# <<<<<<<<<<<<<<<<<<< Read data  >>>>>>>>>>>>>>>>>>
# This is a dictionary with keys being funcs, and values being a dictionary, whose keys are Ns, and values are np.array
# of absolute function values over corresponding distribution.
# {'det' : {6: [...], 7: [...], ...}, 'perm': {6: [...], 7: [...], ...}, ...}

print('Start to read data')
multi_func_raw_data = read_raw_acc_files(funcs, repeats_list)
# multi_func_refactored_data = read_refactored_acc_files(funcs, repeats_list)

print('Read data finished')
# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
plt.rcParams.update({'font.size': 8})
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fontsize = 16
Ns = np.arange(6, 35)
funcs_to_plot = ['det', 'perm', 'haf', 'lhaf_w=0.1', 'lhaf_w=1']

# # <<<<<<<<<<<<<<<<<<< Plot cdf[1/epsilon] against epsilon  >>>>>>>>>>>>>>>>>>
epsilons = np.logspace(start=0, stop=2, num=20, base=10)
np.save(DFUtils.create_filename(plt_dir + rf'\data\epsilons.npy'), epsilons)
for i_N, N in enumerate(Ns):

    print(f'Plot cdf(1/epsilon) against epsilon for N={N}')

    for i_func, func in enumerate(funcs_to_plot):
        func_label = func_labels(func)

        if N in multi_func_raw_data[func].keys():
            # Refactor data
            raw_data = multi_func_raw_data[func][N]
            refactored_data = prefactor(N, func) * raw_data

            cdfs = np.zeros((len(epsilons)), dtype=float)
            error_bars = np.zeros((2, len(epsilons)), dtype=float)

            for i_epsilon, epsilon in enumerate(epsilons):

                print(f'Calculate cdf for {func} and epsilon={epsilon}')
                t1 = time.time()
                cdf, n_error, p_error = get_cdf(refactored_data, 1/epsilon, bootstrapping=bootstrapping)
                t2 = time.time()
                # print(f'Calculate finished after {t2-t1}s. Results={(cdf, n_error, p_error)}')

                cdfs[i_epsilon] = cdf
                error_bars[:, i_epsilon] = np.array([n_error, p_error])

            np.save(DFUtils.create_filename(plt_dir + rf'\data\{func}\N={N}_cdfs.npy'), cdfs)
            np.save(DFUtils.create_filename(plt_dir + rf'\data\{func}\N={N}_errors.npy'), error_bars)

            plt.figure(f'cdf for N={N}')
            plt.errorbar(epsilons, cdfs, fmt='o', ls='none', yerr=error_bars, color=cycle[i_func], label=func_label)

            plt.figure(f'cdf for {func}')
            plt.errorbar(epsilons, cdfs, fmt='o', ls='None', yerr=error_bars, label=f'N={N}')

        else:
            continue

    plt.figure(f'cdf for N={N}')
    plt.xlabel(r'$\epsilon$', fontsize=fontsize)
    plt.ylabel(r'$F(N, 1/\epsilon)$', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize-4)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-6, 2)
    if i_N == 0:
        plt.legend(fontsize=fontsize-4)

    if save_fig:
        plt.savefig(DFUtils.create_filename(plt_dir + rf'\cdf_for_N={N}.pdf'))

for func in funcs_to_plot:
    plt.figure(f'cdf for {func}')
    plt.xlabel(r'$\epsilon$', fontsize=fontsize)
    plt.ylabel(r'$F(N, 1/\epsilon)$', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize-4)
    # plt.xticks([0, 1e6])
    plt.yscale('linear')
    plt.xscale('linear')
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 100)
    if func == 'det':
        plt.legend(fontsize=fontsize-6)
    if save_fig:
        plt.savefig(DFUtils.create_filename(plt_dir + rf'\cdf_for_{func}.pdf'))


# # <<<<<<<<<<<<<<<<<<< Plot 1-cdf against N  >>>>>>>>>>>>>>>>>>
special_eps_id = [0, 3, 6]

for eps_id in special_eps_id:
    epsilon = epsilons[eps_id]

    plt.figure(rf'1-cdf for eps={epsilon}')

    for func in funcs_to_plot:
        inv_cdfs = []
        for N in Ns:
            if N in multi_func_raw_data[func].keys():
                cdfs = np.load(plt_dir + rf'\data\{func}\N={N}_cdfs.npy')
                error_bars = np.load(plt_dir + rf'\data\{func}\N={N}_errors.npy')

                inv_cdfs.append({
                    'N': N,
                    '1-cdf': 1 - cdfs[eps_id],
                    'n_error': error_bars[1, eps_id],
                    'p_error': error_bars[0, eps_id]
                })

            else:
                continue

        inv_cdfs_df = pd.DataFrame(inv_cdfs)
        inv_cdfs_df.to_csv(DFUtils.create_filename(plt_dir + rf'\data\{func}\1-cdf(N, eps={epsilon}).csv'))

        plt.errorbar(inv_cdfs_df['N'], np.log(list(inv_cdfs_df['1-cdf'])), yerr=[inv_cdfs_df['n_error'], inv_cdfs_df['p_error']],
                     capsize=1, label=func_labels(func), ls='None', fmt='x')

    plt.legend(fontsize=fontsize - 4, loc='lower left')
    plt.ylabel(rf'$\ln(1-F(N, {{{1/epsilon:.2f}}})$', fontsize=fontsize)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    plt.savefig(plt_dir + rf'\1-cdf(N, eps={epsilon:.2f}).pdf')
    plt.ylim(-6, -0.1)


# # <<<<<<<<<<<<<<<<<<< Bootstrap mean  >>>>>>>>>>>>>>>>>>
for i_func, func in enumerate(funcs_to_plot):

    results_rows = []
    for i_N, N in enumerate(Ns):

        if N in multi_func_raw_data[func].keys():
            # Refactor data
            raw_data = multi_func_raw_data[func][N]
            refactored_data = prefactor(N, func) * raw_data

            refactored_mean = np.mean(refactored_data)

            t1 = time.time()
            bootstrap_res = bootstrap((refactored_data,), np.mean, confidence_level=0.9, batch=100, method='basic')
            t2 = time.time()
            print(f'Bootstrapping {func} for N={N} finished after {t2-t1}s')

            results_rows.append({'N': N, 'mean': refactored_mean,
                                 'n_error': refactored_mean - bootstrap_res.confidence_interval.low,
                                 'p_error': bootstrap_res.confidence_interval.high - refactored_mean}
                                )
        else:
            continue

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(DFUtils.create_filename(plt_dir + rf'\data\{func}\bootstrapping_df.csv'), index=False)

    plt.figure('Mean')
    plt.errorbar(results_df['N'], results_df['mean'], yerr=[results_df['n_error'], results_df['p_error']], capsize=5, label=func_labels(func))

fig=plt.figure('Mean')
plt.legend(fontsize=fontsize-4, loc='upper right')
plt.ylim([0, 1])
plt.xlabel('$N$', fontsize=fontsize)
plt.ylabel('Refactored mean', fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize-4)
fig.set_size_inches(10, 6, forward=True)
plt.savefig(plt_dir + rf'\means.pdf')