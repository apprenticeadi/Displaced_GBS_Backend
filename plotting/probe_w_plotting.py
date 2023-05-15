import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import pi
from scipy import stats

from src.utils import DFUtils


def gaussian_fit(x, amp, x0, sigma):
    return amp * np.exp(- (x - x0) ** 2 / (2 * sigma) ** 2)

# def inverse_gaussian(y, amp, x0, sigma):
#     dist_from_x0 = np.sqrt( - np.log(y/amp) * (2 * sigma) ** 2 )
#     return x0 - dist_from_x0, x0 + dist_from_x0

date_time = r'\03-03-2023(20-27-26.624767)'

dir = fr'..\Results\probe_w' + date_time
save_fig = True
plot_dir = dir + r'\plots'
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# This is specific to this simulation. Change it for different simulations. Here I do this to calculate the median.
n_start = 1
n_end = 6
num_n = 101
n_list = np.logspace(n_start, n_end, num=num_n, dtype=int)  # list of n values to test
medians = np.zeros((num_n, 2), dtype=float)
for i, n_i in enumerate(n_list):
    raw = np.load(dir + fr'\N={n_i}.npy')
    medians[i] = np.array([n_i, np.median(raw)])

# Columns are n, min(|Z|), mean(|Z|), std(|Z|)
data_stats = np.load(dir + r'\stats.npy')

hist_n = np.logspace(n_start, n_end, num = n_end-n_start+1, dtype=int)
fig = plt.figure('histograms')
fig.set_size_inches(15, 8)
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fit_results = np.zeros((num_n, 3), dtype=float)  # 'n', 'x0', 'sigma'
for i_n, n in enumerate(n_list):
    Z_abs_values = np.load(dir + fr'\N={n}.npy')

    if np.sum(np.isnan(Z_abs_values)) != 0:
        continue

    mean_z = np.mean(Z_abs_values)
    med_z = np.median(Z_abs_values)

    log_data = np.log(Z_abs_values)
    log_mean_z = np.log(mean_z)
    log_med_z = np.log(med_z)

    if n in hist_n:
        j = np.argmax(hist_n == n)
        heights, bin_edges, _ = plt.hist(log_data, density=True, bins=1000, alpha=0.5, color=cycle[j],
                                     label=f'n=1e{int(np.log10(n))}')
        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        x_length = max_xlim - min_xlim
        # plt.axvline(mean_z, color=cycle[j], linestyle='dashed')
        # plt.text(mean_z*1.1, max_ylim*0.9, f'n=1e{int(np.log10(n))}')

        plt.axvline(log_med_z, color=cycle[j], linestyle='-')
        plt.text(log_med_z * 1.1, max_ylim * 0.9, f'N=1e{int(np.log10(n))}', color=cycle[j])

    else:
        heights, bin_edges = np.histogram(log_data, bins=1000, density=True)


    mid_bins = (bin_edges[1:] + bin_edges[:-1]) / 2

    popt_gauss, pcov_gauss = curve_fit(f=gaussian_fit, xdata=mid_bins, ydata=heights,
                                       p0=[max(heights), log_med_z, np.std(log_data)])

    x0 = popt_gauss[1]
    sigma = popt_gauss[2]

    fit_results[i_n] = np.array([n, x0, sigma])  # some rows are zeros due to some strange stuff with the data

    if n in hist_n:
        plt.plot(mid_bins, gaussian_fit(mid_bins, *popt_gauss), color=cycle[j], linewidth=2, linestyle='--')

        y_line = gaussian_fit(x0 + 3 * sigma, *popt_gauss)
        # plt.axhline(y=y_line, xmin = ((x0 - 3 * sigma) - min_xlim) / x_length, xmax = ((x0 + 3 * sigma) - min_xlim) / x_length, color=cycle[j], linewidth=2)
        plt.axvline(x0 - 3 * sigma, ymin = 0, ymax = y_line / max_ylim, color=cycle[j], linewidth=2)
        plt.axvline(x0 + 3 * sigma, ymin = 0, ymax = y_line / max_ylim, color=cycle[j], linewidth=2)


fit_results = fit_results[np.where(fit_results[:,0] != 0)]  # Get rid of the zero data

plt.xlabel(r'$\ln(|\tilde{X}_{ij}|)$')
plt.ylabel('Probability density function')
# min_xlim, max_xlim = plt.xlim()
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\histograms.png'))

# log_median = np.log(medians[:41, 1])
# log_mean = np.log(data_stats[:41, 2])
# log_min = np.log(data_stats[:41, 3])
# log_ns = np.log(data_stats[:41, 0])

plt.figure('mean value of |tildeX|')
plt.plot(data_stats[:, 0], data_stats[:, 2], 'x', label=r'mean$(|\tilde{X}_{ij}|)$')
plt.plot(medians[:, 0], medians[:, 1], 'x', label=r'median($|\tilde{X}_{ij}|$)')
plt.plot(data_stats[:, 0], data_stats[:, 1], 'x', label=r'min($|\tilde{X}_{ij}|$)')

fit_maxs = np.exp(fit_results[:, 1] + 3 * fit_results[:, 2])
plt.plot(fit_results[:, 0], fit_maxs, 'x', label=r'$3\sigma$ maximum')
log_ns = np.log(fit_results[:,0])
log_max = np.log(fit_maxs)
regress_result = stats.linregress(log_ns, log_max)
slope = regress_result.slope
intercept = regress_result.intercept
plt.plot(n_list, np.exp(intercept) * n_list ** slope, '--', color='black')
plt.text(fit_results[-1,0] / 5, fit_maxs[-1] * 5, fr'{np.exp(intercept):.3}$N^{{{slope:.3}}}$')


plt.plot(fit_results[:, 0], np.exp(fit_results[:, 2]), 'x', label=r'$\exp(\sigma_{log})$')
# plt.plot(data_stats[:, 0], data_stats[:, 3], 'x', label=r'std($|\tilde{X}_{ij}|$)')
plt.plot(data_stats[:, 0], 1 / np.sqrt(data_stats[:, 0]), linestyle='-.', color='black')
plt.text(data_stats[-1, 0], 1 / np.sqrt(data_stats[-1, 0]), r'$1/\sqrt{N}$')
plt.plot(data_stats[:, 0], 1 / data_stats[:, 0], linestyle='--', color='black')
plt.text(data_stats[-1, 0], 1 / data_stats[-1, 0], r'$1/N$')
# plt.plot(data_stats[:, 0], 1 / data_stats[:, 0] ** 2, linestyle='-', color='black', label=r'$1/N^2$')
plt.xlabel(r'$N$')
# plt.title(r'Mean and min of $|\tilde{X}_{ij}|$ against $N$')
# plt.xticks(n_list[::11])
plt.yscale('log')
plt.xscale('log')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\mean_and_min.png'))

# plot std of |Z|
# plt.figure('std of |tildeX|')
# plt.plot(data_stats[:, 0], data_stats[:, 3], 'x', label=r'std($|\tilde{X}_{ij}|$)')
# plt.xlabel('N')
# plt.title(r'Std of $|\tilde{X}_{ij}|$ against N')
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(data_stats[:, 0], 1 / data_stats[:, 0], linestyle='--', color='black', label='1/sqrt(M)')
# plt.legend()
# if save_fig:
#     plt.savefig(DFUtils.create_filename(plot_dir + r'\std.png'))
