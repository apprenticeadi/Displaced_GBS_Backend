import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy.stats import bootstrap
from src.utils import DFUtils, DGBSUtils
import datetime
import time

total_Ns = list(range(4, 17))
reps = 10000

# <<<<<<<<<<<<<<<<<<< Basic parameters  >>>>>>>>>>>>>>>>>>
Ns_dir_dict = {
    rf'..\Results\roots_matching_polynomial\{reps}rep_2023-08-23(18-00-43.689530)': list(range(4, 15)),
    rf'..\Results\roots_matching_polynomial\{reps}rep_2023-08-23(19-35-17.526866)': [15],
    rf'..\Results\roots_matching_polynomial\{reps}rep_2023-08-25(17-08-36.941174)': [16]
}

# Ns_dir_dict = {
#     rf'..\Results\roots_matching_polynomial\{reps}rep_2023-09-10(19-35-56.363960)': list(range(4, 16))
# }

save_fig = False
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plot_dir = fr'..\Plots\roots_mp\tildeX_distrib_{time_stamp}'
# plot_dir = fr'..\Plots\roots_mp\symGauss_distrib_{time_stamp}'
# <<<<<<<<<<<<<<<<<<< Reading raw data  >>>>>>>>>>>>>>>>>>
def find_quart(a, axis=0):
    return np.quantile(a, 0.25, axis=axis)

# Find the min root of each of the 10000 matching polynomials. Bootstrap find their median and first quartile.
min_roots = np.zeros((len(total_Ns), reps), dtype=np.complex128)  # the smallest root of each distinct matching polynomial
abs_min_roots = np.zeros((len(total_Ns), reps), dtype=float)

median_min_roots = np.zeros((len(total_Ns), 3), dtype=float)  # median of min_roots_mp and their confidence interval
quart_min_roots = np.zeros((len(total_Ns), 3), dtype=float)  # first quartile of min_roots_mp and their confidence interval
for results_dir in Ns_dir_dict.keys():
    Ns = Ns_dir_dict[results_dir]
    for N in Ns:
        roots_N = np.load(results_dir + fr'\N={N}_roots.npy')
        i_N = total_Ns.index(N)

        # the min root from each distinct matching polynomial
        roots_N = roots_N.reshape((reps, N // 2))
        row_args = np.abs(roots_N).argmin(axis=1)
        min_roots_N = np.zeros(reps, dtype=np.complex128)
        for i_row, row_arg in enumerate(row_args):
            min_roots_N[i_row] = roots_N[i_row, row_arg]
        min_roots[i_N] = min_roots_N

        abs_roots_N = np.abs(min_roots_N)
        abs_min_roots[i_N] = abs_roots_N

        print(f'Bootstrapping median for N={N}')
        # bootstrapping
        t1 = time.time()
        median = np.median(abs_roots_N)
        med_res = bootstrap((abs_roots_N,), np.median, confidence_level=0.95)
        median_min_roots[i_N, :] = (median, median - med_res.confidence_interval.low, med_res.confidence_interval.high - median)
        t2 = time.time()
        print(f'Bootstrapping finished after {t2-t1}s. Bootstrapping first quartile.')

        quart = find_quart(abs_roots_N)
        quart_res = bootstrap((abs_roots_N,), find_quart, confidence_level=0.95)
        quart_min_roots[i_N, :] = (quart, quart - quart_res.confidence_interval.low, quart_res.confidence_interval.high - quart)
        t3 = time.time()
        print(f'Bootstrapping finished after {t3-t2}s. ')

np.save(DFUtils.create_filename(plot_dir + f'\median_min_roots_mp.npy'), median_min_roots)
np.save(plot_dir + f'\quart_min_roots_mp.npy', quart_min_roots)

# <<<<<<<<<<<<<<<<<<< Plot specs  >>>>>>>>>>>>>>>>>>
log_linthresh = -4
linthresh = 10 ** (log_linthresh)
axis_log_lim = 1
axis_lim = 10 ** (axis_log_lim)
half_axis_ticks = 10 ** np.arange(log_linthresh, axis_log_lim + 1, step=2, dtype=np.float64)
axis_ticks = np.concatenate([-half_axis_ticks, half_axis_ticks, [0]])

# <<<<<<<<<<<<<<<<<<< Plot for each N  >>>>>>>>>>>>>>>>>>
# plot the min root of each matching polynomial
for i_N, N in enumerate(total_Ns):
    save_name = plot_dir + fr'\min_root_of_each_mp_N={N}.pdf'

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in min_roots[i_N]]
    roots_imag = [np.imag(r) for r in min_roots[i_N]]

    plt.figure(f'min root of each mp N={N}')

    # Remove plot boundaries (spines)
    ax = plt.gca()  # Get the current Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Create solid lines for the real and imaginary axes
    plt.axhline(0, color='black', linewidth=1)  # Real axis
    plt.axvline(0, color='black', linewidth=1)  # Imaginary axis

    # Scatter plot roots
    plt.scatter(roots_real, roots_imag, marker='x', s=10)

    # create circular boundary of median roots
    median_root = median_min_roots[i_N, 0]
    circle_median = plt.Circle((0, 0), median_root, color='black', fill=False, label=fr'$|z|={{{median_root:.3f}}}$')
    plt.gca().add_patch(circle_median)

    quart_root = quart_min_roots[i_N, 0]
    circle_quart = plt.Circle((0, 0), quart_root, color='red', fill=False, label=fr'$|z|={{{quart_root:.3f}}}$')
    plt.gca().add_patch(circle_quart)

    plt.xscale('symlog', linthresh=linthresh)
    plt.yscale('symlog', linthresh=linthresh)

    # Set axes
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)

    plt.xlabel(r'Re')
    plt.ylabel(r'Im')

    plt.legend(loc='upper right')

    plt.show()

    if save_fig:
        plt.savefig(DFUtils.create_filename(save_name))
#
# <<<<<<<<<<<<<<<<<<< Plot median w  >>>>>>>>>>>>>>>>>>
plt.figure('medians')

plt.errorbar(total_Ns, median_min_roots[:, 0], yerr=median_min_roots[:, 1:].T, fmt='x', color='black', linestyle='None', label='median')
plt.errorbar(total_Ns, quart_min_roots[:, 0], yerr=quart_min_roots[:, 1:].T, fmt='x', color='red', linestyle='None', label='first quartile')
plt.xlabel(r'$N$')
plt.ylabel(r'|z|')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\medians.pdf'))

# <<<<<<<<<<<<<<<<<<< Plot success probability colour plot  >>>>>>>>>>>>>>>>>>
w_max = 5.000
w_min = 0.0001
ws = np.linspace(w_min, w_max, 10000)
epsilons = 1 / ws ** 2
epsilons = epsilons[::-1]

color_mesh = np.zeros((len(total_Ns), len(epsilons)), dtype=float)
for i_eps, epsilon in enumerate(epsilons):
    counts = np.sum(abs_min_roots <= epsilon, axis=1)
    color_mesh[:, i_eps] = counts / reps

plt.figure('color plot Pr(non zero)')
plt.pcolormesh(np.concatenate([[2 * ws[-1] - ws[-2]], ws[::-1]]), np.arange(min(total_Ns), max(total_Ns) + 2) - 0.5,
               1- color_mesh, vmin=0, vmax=1)
plt.colorbar()
plt.xlabel(r'$|w|$')
plt.ylabel(r'$N$')
plt.xscale('linear')
plt.ylim([min(total_Ns) - 0.5, max(total_Ns) + 0.5])
plt.yticks(total_Ns[::2])
plt.xlim([min(ws), max(ws)])
for N in total_Ns:
    plt.axhline(N - 0.5, xmin=0, xmax=w_max, color='white', linestyle=':')

# Errorbar plots of median and first quantile
med_ws = 1 / np.sqrt(median_min_roots[:, 0])
med_ws_err = np.zeros((2, len(med_ws)), dtype=float)
for i_N in range(len(median_min_roots)):
    med_ws_err[0, i_N] = 0.5 * median_min_roots[i_N, 1] / median_min_roots[i_N, 0] * med_ws[i_N]  # lower error
    med_ws_err[1, i_N] = 0.5 * median_min_roots[i_N, 2] / median_min_roots[i_N, 0] * med_ws[i_N] # upper error
plt.errorbar(med_ws, total_Ns, xerr=med_ws_err, fmt='x', color='black', linestyle='None')  # this is 50%

quart_ws = 1 / np.sqrt(quart_min_roots[:, 0])
quart_ws_err = np.zeros((2, len(quart_ws)), dtype=float)
for i_N in range(len(quart_min_roots)):
    quart_ws_err[0, i_N] = 0.5 * quart_min_roots[i_N, 1] / quart_min_roots[i_N, 0] * quart_ws[i_N] # lower error
    quart_ws_err[1, i_N] = 0.5 * quart_min_roots[i_N, 2] / quart_min_roots[i_N, 0] * quart_ws[i_N] # upper error
plt.errorbar(quart_ws, total_Ns, xerr=quart_ws_err, fmt='x', color='red', linestyle='None')  # this is 25%

if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\color_mesh_prob_non_zero.pdf'))

# <<<<<<<<<<<<<<<<<<< Plot photon number ratio  >>>>>>>>>>>>>>>>>>
#
#
# def solve_photon_ratio(w, lower_err, upper_err):
#
#     r, beta = DGBSUtils.solve_w(w, N_mean=1)
#     ratio = beta**2 / np.sinh(r)**2
#
#     r1, beta1 = DGBSUtils.solve_w(w-lower_err, N_mean=1)
#     ratio1 = beta1**2 / np.sinh(r1)**2
#
#     r2, beta2 = DGBSUtils.solve_w(w+upper_err, N_mean=1)
#     ratio2 = beta2**2 / np.sinh(r2)**2
#
#     return ratio, ratio-ratio1, ratio2-ratio
#
# # solve for photon ratio
# med_ratios = np.zeros((len(total_Ns), 3), dtype=float) # include error
# quart_ratios = np.zeros((len(total_Ns), 3), dtype=float)
# for i_N, N in enumerate(total_Ns):
#     med_ratios[i_N, :] = solve_photon_ratio(med_ws[i_N], med_ws_err[0, i_N], med_ws_err[1, i_N])
#     quart_ratios[i_N, :] = solve_photon_ratio(quart_ws[i_N], quart_ws_err[0, i_N], quart_ws_err[1, i_N])
#
# plt.figure('Photon ratios')
# plt.errorbar(total_Ns, med_ratios[:, 0], yerr=med_ratios[:, 1:].T, fmt='x', color='black', linestyle='None')
# plt.errorbar(total_Ns, quart_ratios[:, 0], yerr=quart_ratios[:, 1:].T, fmt='x', color='red', linestyle='None')
# plt.ylabel(r'$|\beta|^2/\sinh^2(r)$')
# plt.xlabel(r'$N$')
#
# if save_fig:
#     plt.savefig(DFUtils.create_filename(plot_dir + r'\photon_ratios.pdf'))


# # <<<<<<<<<<<<<<<<<<< Plot subfigures for writeup  >>>>>>>>>>>>>>>>>>
fontsize = 18
# fig = plt.figure('cluster truncation success prob', layout='constrained', figsize=(6,9))
# gs = fig.add_gridspec(18, 12)
# # subfigs=fig.subfigures(2, 1)
# # axup = subfigs[0].subplots(1, 1)
# axup = fig.add_subplot(gs[1:12, :-1])
# axbottom = fig.add_subplot(gs[12:-1, :-1])
# pc = axup.pcolormesh(np.concatenate([[2 * ws[-1] - ws[-2]], ws[::-1]]), np.arange(min(total_Ns), max(total_Ns) + 2) - 0.5, 1 - color_mesh, vmin=0, vmax=1)
# # subfigs[0].colorbar(pc, shrink=0.6, ax=axup, location='bottom', aspect=30)
# cb = plt.colorbar(pc, ax=axup, location='bottom', shrink=0.8, aspect=20)
# cb.ax.tick_params(labelsize=fontsize-2)
#
# axup.set_xlabel(r'$|w|$', fontsize=fontsize)
# axup.set_ylabel(r'$N$', fontsize=fontsize)
# axup.set_ylim([min(total_Ns) - 0.5, max(total_Ns) + 0.5])
# axup.set_yticks(total_Ns[::2], fontsize=fontsize)
# axup.set_xticks(np.arange(5)+1, fontsize=fontsize)
# axup.tick_params(axis='both', which='major', labelsize=fontsize)
#
# axup.set_xlim([min(ws), max(ws)])
# axup.set_title('(a)', loc='left')
# for N in total_Ns:
#     axup.axhline(N - 0.5, xmin=0, xmax=w_max, color='white', linestyle=':')
# axup.errorbar(med_ws, total_Ns, xerr=med_ws_err, fmt='x', color='black', linestyle='None')  # this is 50%
# axup.errorbar(quart_ws, total_Ns, xerr=quart_ws_err, fmt='x', color='red', linestyle='None')  # this is 25%
#
# # axbottom = subfigs[1].subplots(1,1)
#
# axbottom.errorbar(total_Ns, med_ratios[:, 0], yerr=med_ratios[:, 1:].T, fmt='x', color='black', linestyle='None')
# axbottom.errorbar(total_Ns, quart_ratios[:, 0], yerr=quart_ratios[:, 1:].T, fmt='x', color='red', linestyle='None')
# axbottom.set_ylabel(r'$|\beta|^2/\sinh^2(r)$', fontsize=fontsize)
# axbottom.set_xlabel(r'$N$', fontsize=fontsize)
# axbottom.set_xticks(total_Ns[::2])
# axbottom.set_yticks([50, 100, 150, 200, 250])
# axbottom.tick_params(axis='both', which='major', labelsize=fontsize)
# axbottom.set_title('(b)', loc='left')
# axbottom.set_ylim([0, 250])
#
# if save_fig:
#     plt.savefig(DFUtils.create_filename(plot_dir + r'\cluster_truncation_success_probs.svg'))


# <<<<<<<<<<<<<<<<<<< Plot root distribution subplots   >>>>>>>>>>>>>>>>>>
axis_ticks = np.array([-1., -1e-3, 0, 1e-3, 1])
spine_alpha = 0.5
axis_alpha = 0.8

fig = plt.figure('Root distribution subplots', layout='constrained')
gs = fig.add_gridspec(2,2, hspace=0.1, wspace=0.1)
axs = gs.subplots(sharex='col', sharey='row').flatten()
N_toplot = np.array([4, 8, 12, 15])
title_label = ['(a)', '(b)', '(c)', '(d)']

scatter_paths = []
for i_N_t, N in enumerate(N_toplot):

    i_N = total_Ns.index(N)
    ax = axs[i_N_t]

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in min_roots[i_N]]
    roots_imag = [np.imag(r) for r in min_roots[i_N]]

    # spines less transparent
    ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(spine_alpha)

    # # Create solid lines for the real and imaginary axes
    ax.axhline(0, color='black', linewidth=1, alpha=axis_alpha)  # Real axis
    ax.axvline(0, color='black', linewidth=1, alpha=axis_alpha)  # Imaginary axis

    # Scatter plot roots
    dots = ax.scatter(roots_real, roots_imag, marker='o', s=6, alpha=0.2)
    scatter_paths.append(dots)

    # create circular boundary of median roots
    median_root = median_min_roots[i_N, 0]
    circle_median = patches.Circle((0, 0), median_root, color='black', fill=False, label=fr'$|z|={{{median_root:.3f}}}$')
    ax.add_patch(circle_median)

    quart_root = quart_min_roots[i_N, 0]
    circle_quart = patches.Circle((0, 0), quart_root, color='red', fill=False, label=fr'$|z|={{{quart_root:.3f}}}$')
    ax.add_patch(circle_quart)

    print(fr'N={N}, median_root={median_root}, quart_root={quart_root}')

    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('symlog', linthresh=linthresh)

    # Set axes
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 6)

    # ax.set_xlabel(f'N={N}', fontsize=fontsize-2)

    ax.set_title(fr'{title_label[i_N_t]} N={N}', fontsize=fontsize-2, loc='left')

fig.supxlabel(r'Re$(z)$', fontsize=fontsize)
fig.supylabel(r'Im$(z)$', fontsize=fontsize)

if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\min_root_subfigs.pdf'))