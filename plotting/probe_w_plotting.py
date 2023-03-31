import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.utils import DFUtils

date_time = r'\03-03-2023(20-27-26.624767)'

dir = fr'..\Results\probe_w'+date_time
save_fig = False
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
stats = np.load(dir + r'\stats.npy')

hist_n = np.logspace(1, 5, num=5, dtype=int)
fig = plt.figure('histograms')
fig.set_size_inches(15, 8)
bins = np.logspace(-5, 3, num=1000)
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for j, n in enumerate(hist_n):


    Z_abs_values = np.load(dir + fr'\N={n}.npy')

    mean_z = np.mean(Z_abs_values)
    med_z = np.median(Z_abs_values)

    plt.hist(Z_abs_values, weights=np.ones_like(Z_abs_values) / Z_abs_values.size, bins=bins, log=True, alpha=0.5, color=cycle[j], label=f'n=1e{int(np.log10(n))}')

    min_ylim, max_ylim = plt.ylim()
    plt.axvline(mean_z, color=cycle[j], linestyle='dashed')
    # plt.text(mean_z*1.1, max_ylim*0.9, f'n=1e{int(np.log10(n))}')

    plt.axvline(med_z, color=cycle[j], linestyle='-')
    plt.text(med_z*1.1, max_ylim*0.9, f'N=1e{int(np.log10(n))}', color=cycle[j])

plt.xscale('log')
plt.xlabel(r'$|\tilde{X}_{ij}|$')
plt.ylabel('Frequency')
min_xlim, max_xlim = plt.xlim()
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\histograms.png'))

# log_median = np.log(medians[:41, 1])
# log_mean = np.log(stats[:41, 2])
# log_min = np.log(stats[:41, 3])
# log_ns = np.log(stats[:41, 0])

plt.figure('mean value of |tildeX|')
plt.plot(stats[:, 0], stats[:, 2], 'x', label=r'mean$(|\tilde{X}_{ij}|)$')
slope, intercept, r_value, p_value, std_err = stats.linregress(log_ns, log_mean)
plt.plot(medians[:, 0], medians[:, 1], 'x', label=r'median($|\tilde{X}_{ij}|$)')
plt.plot(stats[:, 0], stats[:, 1], 'x', label=r'min($|\tilde{X}_{ij}|$)')
plt.plot(stats[:, 0], stats[:, 3], 'x', label=r'std($|\tilde{X}_{ij}|$)')
plt.plot(stats[:, 0], 1 / np.sqrt(stats[:, 0]), linestyle='-.', color='black', label=r'$1/\sqrt{N}$')
plt.plot(stats[:, 0], 1 / stats[:, 0], linestyle='--', color='black', label=r'$1/N$')
# plt.plot(stats[:, 0], 1 / stats[:, 0] ** 2, linestyle='-', color='black', label=r'$1/N^2$')
plt.xlabel(r'$N$')
plt.title(r'Mean and min of $|\tilde{X}_{ij}|$ against $N$')
# plt.xticks(n_list[::11])
plt.yscale('log')
plt.xscale('log')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\mean_and_min.png'))

# plot std of |Z|
# plt.figure('std of |tildeX|')
# plt.plot(stats[:, 0], stats[:, 3], 'x', label=r'std($|\tilde{X}_{ij}|$)')
# plt.xlabel('N')
# plt.title(r'Std of $|\tilde{X}_{ij}|$ against N')
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(stats[:, 0], 1 / stats[:, 0], linestyle='--', color='black', label='1/sqrt(M)')
# plt.legend()
# if save_fig:
#     plt.savefig(DFUtils.create_filename(plot_dir + r'\std.png'))