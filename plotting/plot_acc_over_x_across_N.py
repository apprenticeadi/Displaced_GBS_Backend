import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
import os
import datetime

from src.utils import DFUtils
from src.photon_number_distributions import big_F


# <<<<<<<<<<<<<<<<<<< Functions  >>>>>>>>>>>>>>>>>>
def prefactor(_N, _func):
    if _func == 'det' or _func == 'perm':
        return 1 / np.sqrt(factorial(_N))
    elif _func == 'haf':
        if _N % 2 == 0:
            return 1 / np.sqrt(comb(_N - 1, _N / 2) * factorial(_N))
        else:
            return 0

    elif _func[:4] == 'lhaf':
        if _func[_func.index('w') + 2:] == "N^-1":  # currently only recognizes 1/N
            w = 1 / _N
        else:
            w = float(_func[_func.index('w') + 2:])
        # return np.sqrt(np.power(2., float(_N)) / factorial(_N) / big_F(w, _N, _N)[_N])
        return np.sqrt( np.power(2., float(_N)) * comb(_N**2, _N) / big_F(w, _N, _N)[_N] / np.power(float(_N), 2*float(_N)) )

    else:
        raise ValueError(f'{_func} not recognized')


def get_cdf(data_arr, xs):
    """cumulative distribution function"""
    data_arr = np.atleast_2d(data_arr)
    num_N, num_repeats = data_arr.shape

    cdf = np.zeros((num_N, len(xs)))

    for i, x in enumerate(xs):
        cdf[:, i] = np.argmax(data_arr > x, axis=1) / num_repeats

    return cdf


def read_acc_files(funcs, dir_head, repeats, cdf_density=16):
    all_func_dirs = os.listdir(dir_head)

    processed_data = {}
    for func in funcs:

        _func_dirs = [file_ for file_ in all_func_dirs if
                      file_.startswith(func + '_')]  # add '_' otherwise w=10 and w=1 get mixed up

        N_list = []
        raw_data_list = []
        for _func_dir in _func_dirs:
            for N_str in os.listdir(dir_head + fr'\{_func_dir}'):
                N = int(N_str[2:])
                if N in N_list:
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

                N_list.append(N)
                raw_data_list.append(combined_raw_data)

        N_arr = np.asarray(N_list)
        raw_data_arr = np.asarray(raw_data_list)

        raw_data_arr = raw_data_arr[np.argsort(N_arr)]
        N_arr.sort()

        refactored_data_arr = np.zeros((len(N_arr), repeats))
        for j, N in enumerate(N_arr):
            refactored_raw_data = prefactor(N, func) * raw_data_arr[j]
            refactored_data_arr[j, :] = refactored_raw_data

        x_min = np.min(refactored_data_arr)
        x_min_log = np.floor(np.log10(x_min))
        x_max_log = 0

        xs = np.logspace(start=x_min_log, stop=x_max_log, num=cdf_density * int(x_max_log - x_min_log), base=10 )  # evenly distributed in log scale, 16 per decade


        cum_distrib_arr = get_cdf(refactored_data_arr, xs)

        processed_data[func] = {
            'Ns': N_arr,
            'xs': xs,
            'raw_data': raw_data_arr,
            'refactored_data': refactored_data_arr,
            'cdf': cum_distrib_arr
        }

    return processed_data


def func_labels(func_string):
    if func_string == 'det':
        return 'Det'
    elif func_string == 'perm':
        return 'Per'
    elif func_string == 'haf':
        return 'Haf'
    elif func_string[:4] == 'lhaf':
        if func_string[func_string.index('w') + 2:] == 'N^-1':
            return r'lHaf$(w=\frac{1}{N})$'
        else:
            w_value = func_string[func_string.index('w') + 2:]
            return fr'lHaf$(w={w_value})$'
    else:
        raise ValueError(f'{func_string} not supported')


# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
repeats = 100000
# repeats = 5000

dir_head = fr'..\Results\anticoncentration_over_X\{repeats}repeats'

# funcs = ['perm', 'det', 'haf', 'lhaf_w=N^-1', 'lhaf_w=0.1', 'lhaf_w=0.01', 'lhaf_w=1']
funcs = ['perm', 'det', 'haf', 'lhaf_w=0.1', 'lhaf_w=1']
# funcs = ['lhaf_w=1', 'lhaf_w=0.1']


# xs_toplot = [1., 0.75, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13]  # for cumulative distribution function
xs_toplot = [1., 0.42, 0.13]

save_fig = False
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
plt_dir = fr'..\Plots\acc_numerics\{time_stamp}'

# <<<<<<<<<<<<<<<<<<< Read data  >>>>>>>>>>>>>>>>>>
processed_data = read_acc_files(funcs, dir_head, repeats, cdf_density=16)

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
plt.rcParams.update({'font.size':14})
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i_func, func in enumerate(funcs):
    refactored_data = processed_data[func]['refactored_data']
    raw_data = processed_data[func]['raw_data']
    Ns = processed_data[func]['Ns']

    # # <<<<<<<<<<<<<<<<<<< Plot distribution of refactored values for each function  >>>>>>>>>>>>>>>>>>
    # plt.figure(f'{func}')
    # for j, N in enumerate(Ns):
    #
    #     if j % 4 == 0:  # Only plot every 4 of them
    #
    #         plt.plot(list(range(repeats)), refactored_data[j, :], label=fr'$N={N}$')
    #
    # x_min, x_max = plt.xlim()
    # plt.axhline(xs_toplot[0], xmin=x_min, xmax=x_max, color='black', linestyle='-')
    # plt.axhline(xs_toplot[-1], xmin=x_min, xmax=x_max, color='black', linestyle='--')
    # plt.text(x_min, xs_toplot[0] * 1.1, fr'$\epsilon={xs_toplot[0]}$')
    # plt.text(x_min, xs_toplot[-1] * 1.1, fr'$\epsilon={xs_toplot[-1]:.3}$')
    # plt.ylabel(r'$\epsilon$')
    # plt.xlabel('trials')
    # plt.legend()
    # plt.yscale('log')
    # plt.xticks([1, repeats])
    # plt.title(rf'distribution of values for |{func_labels(func)}|')
    #
    # if save_fig:
    #     plt.savefig(DFUtils.create_filename(plt_dir + fr'\{func}.png'))
    #
    # # <<<<<<<<<<<<<<<<<<< Plot mean of raw values against 1/prefactor  >>>>>>>>>>>>>>>>>
    # plt.figure(f'{func} mean scaling with prefactor')
    # plt.plot(Ns, np.mean(raw_data, axis=1), label='raw data mean')
    # plt.plot(Ns, np.median(raw_data, axis=1), label='raw data median')
    # inv_prefactors = np.zeros_like(Ns, dtype=float)
    # for j, N in enumerate(Ns):
    #     inv_prefactors[j] = 1 / prefactor(N, func)
    # plt.plot(Ns, inv_prefactors, label='inverse prefactor')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.xlabel(r'$N$')
    # if save_fig:
    #     plt.savefig(DFUtils.create_filename(plt_dir + fr'\{func} mean and prefactor scaling against N.png'))
    #
    # # <<<<<<<<<<<<<<<<<<< For different functions, plot F(N,epsilon) against N for different epsilon >>>>>>>>>>>>>>>>>>
    # plt.figure(fr'{func} $F(N,\epsilon)$')

    cum_distrib = get_cdf(refactored_data, xs_toplot)

    # for i, x in enumerate(xs_toplot):
    #     plt.plot(Ns, cum_distrib[:, i], label=fr'$\epsilon={x:.3}$')
    #
    # plt.legend()
    # plt.xlabel(r'$N$')
    # plt.ylabel(r'$F(N, \epsilon)$')
    # plt.xticks(Ns)
    # plt.xscale('linear')
    # plt.ylim(0, 1)
    # plt.title(rf'$F(N,\alpha)$ for {func_labels(func)}')
    # if save_fig:
    #     plt.savefig(DFUtils.create_filename(plt_dir + fr'\{func} F(N, x) against N.png'))


    # <<<<<<<<<<<<<<<<<<< Plot F(N, epsilon) against N for different functions  >>>>>>>>>>>>>>>>>>
    x_id = -1
    x_special = xs_toplot[x_id]
    plt.figure(fr'$F(N, \alpha={x_special:.3})$ for different functions')

    plt.plot(Ns, 1 - cum_distrib[:, x_id], color=cycle[i_func])
    plt.text(Ns[-1]+1, 0.9 - 0.9 * cum_distrib[-1, x_id], func_labels(func), color=cycle[i_func] )

plt.figure(fr'$F(N, \alpha={x_special:.3})$ for different functions')
plt.xlabel(r'$N$')
plt.ylabel(fr'$1-F(N,\alpha={x_special:.3})$')
plt.xscale('linear')
plt.xlim([0, 55])
plt.yscale('log')
plt.figure(fr'$F(N, \alpha={x_special:.3})$ for different functions').set_figwidth(8)
plt.figure(fr'$F(N, \alpha={x_special:.3})$ for different functions').set_figheight(5)
# plt.legend()
# plt.title(fr'$1 - F(N, \alpha={x_special})$ for different functions')

if save_fig:
    plt.savefig(plt_dir + fr'\F(N, {x_special:.3}) against N.png')

# # <<<<<<<<<<<<<<<<<<< Plot F(N, x) against x for different functions  >>>>>>>>>>>>>>>>>>
# N_special = 22
# plt.figure(fr'$F(N={N_special}, x)$ for different functions')
# for func in funcs:
#     N_id = np.argmax(processed_data[func]['Ns'] == N_special)
#     plt.plot(processed_data[func]['xs'], processed_data[func]['cdf'][N_id, :], '.', label=func_labels(func))
#
# plt.xlabel(r'$x$')
# plt.title(fr'$F(N={N_special}, x)$ for different functions')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# if save_fig:
#     plt.savefig(plt_dir + fr'\F({N_special}, x) against x.png')

# # <<<<<<<<<<<<<<<<<<< Plot 10^5 function values for different functions  >>>>>>>>>>>>>>>>>>
# plt.figure(f'function values for N={N_special}')
# for func in funcs:
#     refactored_data = processed_data[func]['refactored_data']
#     N_id = np.argmax(processed_data[func]['Ns'] == N_special)
#     plt.plot(list(range(repeats)), refactored_data[N_id, :], label=func_labels(func))
#
# x_min, x_max = plt.xlim()
# plt.axhline(xs_toplot[0], xmin=x_min, xmax=x_max, color='black', linestyle='-')
# plt.axhline(xs_toplot[-1], xmin=x_min, xmax=x_max, color='black', linestyle='--')
# plt.text(x_min, xs_toplot[0] * 1.1, fr'$\epsilon={xs_toplot[0]}$')
# plt.text(x_min, xs_toplot[-1] * 1.1, fr'$\epsilon={xs_toplot[-1]:.3}$')
# plt.ylabel(r'$\epsilon$')
# plt.xlabel('trials')
# plt.legend()
# plt.yscale('log')
# plt.xticks([1, repeats])
# plt.title(rf'Distribution of values at N={N_special}')
# if save_fig:
#     plt.savefig(DFUtils.create_filename(plt_dir + fr'\different functions at N={N_special}.png'))

