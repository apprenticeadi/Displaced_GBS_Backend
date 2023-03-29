import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
import os

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
        return np.sqrt(np.power(2., float(_N)) / factorial(_N) / big_F(w, _N, _N)[_N])

    else:
        raise ValueError(f'{_func} not recognized')


def get_cdf(data_arr, xs):
    """cumulative distribution function"""
    data_arr = np.atleast_2d(data_arr)
    num_N, num_repeats = data_arr.shape

    cdf = np.zeros((num_N, len(xs)))

    for i, x in enumerate(xs):

        cdf[:, i] = np.argmax(data_arr>x, axis=1) / num_repeats

    return cdf

def read_acc_files(funcs, dir_head, repeats, cdf_density=16):
    all_func_dirs = os.listdir(dir_head)

    processed_data = {}
    for func in funcs:

        _func_dirs = [file_ for file_ in all_func_dirs if file_.startswith(func)]

        N_list = []
        raw_data_list = []
        for _func_dir in _func_dirs:
            for N_str in os.listdir(dir_head + fr'\{_func_dir}'):
                N = int(N_str[2:])
                if N in N_list:
                    continue

                #TODO: fix this. this is a temporary solution for a half-run permanent calculation
                if func=='perm' and N==26:
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
            refactored_data_arr[j,:] = refactored_raw_data

        x_min = np.min(refactored_data_arr)
        x_min_log = np.floor(np.log10(x_min))
        x_max_log = 0
        xs = np.logspace(start=x_min_log, stop=10**x_max_log, num=cdf_density * int(x_max_log - x_min_log) )  # evenly distributed in log scale, 16 per decade

        cum_distrib_arr = get_cdf(refactored_data_arr, xs)

        processed_data[func]={
            'Ns': N_arr,
            'xs': xs,
            'raw_data': raw_data_arr,
            'refactored_data': refactored_data_arr,
            'cdf': cum_distrib_arr
        }

    return processed_data

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
repeats = 100000

dir_head = fr'..\Results\anticoncentration_over_X\{repeats}repeats'

funcs = ['det', 'haf', 'lhaf_w=1', 'lhaf_w=N^-1', 'perm']
# funcs = ['haf']
func_labels = {
    'det': r'Det',
    'perm': r'Per',
    'haf': r'Haf',
    'lhaf_w=1': r'lHaf$(w=1)$',
    'lhaf_w=N^-1': r'lHaf$(w=\frac{1}{N})$'
}

# xs_toplot = [1., 0.75, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13]  # for cumulative distribution function
xs_toplot = [1., 0.42, 0.13]

save_fig = True
plt_dir = fr'..\Plots\acc_numerics'


# <<<<<<<<<<<<<<<<<<< Read data  >>>>>>>>>>>>>>>>>>
processed_data = read_acc_files(funcs, dir_head, repeats, cdf_density=16)

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
for func in funcs:
    refactored_data = processed_data[func]['refactored_data']
    Ns = processed_data[func]['Ns']

    # <<<<<<<<<<<<<<<<<<< Plot distribution of refactored values for each function  >>>>>>>>>>>>>>>>>>
    plt.figure(f'{func}')
    for j, N in enumerate(Ns):

        if j % 4 == 0:  # Only plot every 4 of them
            plt.plot(list(range(repeats)), refactored_data[j,:], label=fr'$N={N}$')
    x_min, x_max = plt.xlim()
    plt.axhline(xs_toplot[0], xmin=x_min, xmax=x_max, color='black', linestyle='-')
    plt.axhline(xs_toplot[-1], xmin=x_min, xmax=x_max, color='black', linestyle='--')
    plt.text(x_min, xs_toplot[0] * 1.1, fr'$\epsilon={xs_toplot[0]}$')
    plt.text(x_min, xs_toplot[-1] * 1.1, fr'$\epsilon={xs_toplot[-1]:.3}$')
    plt.ylabel(r'$\epsilon$')
    plt.xlabel('trials')
    plt.legend()
    plt.yscale('log')
    plt.xticks([1, repeats])
    plt.title(rf'distribution of values for |{func_labels[func]}|')
    if save_fig:
        plt.savefig(DFUtils.create_filename(plt_dir + fr'\{func}.png'))

    # <<<<<<<<<<<<<<<<<<< For different functions, plot F(N,epsilon) against N for different epsilon >>>>>>>>>>>>>>>>>>
    plt.figure(fr'{func} $F(N,\epsilon)$')

    cum_distrib = get_cdf(refactored_data, xs_toplot)

    for i, x in enumerate(xs_toplot):
        plt.plot(Ns, cum_distrib[:, i], label=fr'$\epsilon={x:.3}$')

    plt.legend()
    plt.xlabel(r'$N$')
    plt.ylabel(r'$F(N, \epsilon)$')
    plt.xticks(Ns)
    plt.xscale('linear')
    plt.ylim(0, 1)
    plt.title(rf'$F(N,\epsilon)$ for {func_labels[func]}')
    if save_fig:
        plt.savefig(plt_dir + fr'\{func} F(N, x) against N.png')

    # <<<<<<<<<<<<<<<<<<< Plot F(N, epsilon) against N for different functions  >>>>>>>>>>>>>>>>>>
    x_id = -1
    x_special = xs_toplot[x_id]
    plt.figure(fr'$F(N, \epsilon={x_special:.3})$ for different functions')

    plt.plot(Ns, 1 - cum_distrib[:, x_id], label=func_labels[func])

plt.figure(fr'$F(N, \epsilon={x_special:.3})$ for different functions')
plt.xlabel(r'$N$')
plt.ylabel(r'$1-F(N,\epsilon)$')
plt.xscale('linear')
plt.yscale('log')
plt.legend()
plt.title(fr'$1 - F(N, \epsilon={x_special})$ for different functions')
if save_fig:
    plt.savefig(plt_dir + fr'\F(N, {x_special:.3}) against N.png')

# <<<<<<<<<<<<<<<<<<< Plot F(N, x) against x for different functions  >>>>>>>>>>>>>>>>>>
N_special = 22
plt.figure(fr'$F(N={N_special}, x)$ for different functions')
for func in funcs:
    N_id = np.argmax(processed_data[func]['Ns']==N_special)
    plt.plot(processed_data[func]['xs'], processed_data[func]['cdf'][N_id, :], '.', label=func_labels[func])
plt.xlabel(r'$x$')
plt.title(fr'$F(N={N_special}, x)$ for different functions')
plt.xscale('log')
plt.yscale('log')
plt.legend()
if save_fig:
    plt.savefig(plt_dir + fr'\F({N_special}, x) against x.png')


# <<<<<<<<<<<<<<<<<<< Plot 10^5 function values for different functions  >>>>>>>>>>>>>>>>>>
plt.figure(f'function values for N={N_special}')
for func in funcs:
    refactored_data = processed_data[func]['refactored_data']
    N_id = np.argmax(processed_data[func]['Ns'] == N_special)
    plt.plot(list(range(repeats)), refactored_data[N_id,:], label=func_labels[func])

x_min, x_max = plt.xlim()
plt.axhline(xs_toplot[0], xmin=x_min, xmax=x_max, color='black', linestyle='-')
plt.axhline(xs_toplot[-1], xmin=x_min, xmax=x_max, color='black', linestyle='--')
plt.text(x_min, xs_toplot[0] * 1.1, fr'$\epsilon={xs_toplot[0]}$')
plt.text(x_min, xs_toplot[-1] * 1.1, fr'$\epsilon={xs_toplot[-1]:.3}$')
plt.ylabel(r'$\epsilon$')
plt.xlabel('trials')
plt.legend()
plt.yscale('log')
plt.xticks([1, repeats])
plt.title(rf'Distribution of values at N={N_special}')
if save_fig:
    plt.savefig(DFUtils.create_filename(plt_dir + fr'\different functions at N={N_special}.png'))