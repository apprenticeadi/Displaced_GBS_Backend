import matplotlib.pyplot as plt
import numpy as np
from src.utils import DFUtils

results_dir = r'..\Results\roots_matching_polynomial\10000rep_2023-08-25(17-08-36.941174)'
Ns = [16] # np.arange(4, 15, dtype=int)
save_fig = True


roots_dict = {}
min_roots = np.zeros(len(Ns), dtype=float)
max_roots = np.zeros(len(Ns), dtype=float)
median_roots = np.zeros(len(Ns), dtype=float)
for i_N, N in enumerate(Ns):
    roots_N = np.load(results_dir + fr'\N={N}_roots.npy')
    abs_roots_N = np.abs(roots_N)

    roots_dict[N] = roots_N

    min_roots[i_N] = np.min(abs_roots_N)
    max_roots[i_N] = np.max(abs_roots_N)
    median_roots[i_N] = np.median(abs_roots_N)

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
log_linthresh = int(np.log10(np.min(min_roots))) - 1
linthresh = 10 ** (log_linthresh)
axis_log_lim = int(np.log10(np.max(max_roots))) + 1
axis_lim = 10 ** (axis_log_lim)
half_axis_ticks = 10 ** np.arange(log_linthresh, axis_log_lim+1, step=2, dtype=np.float64)
axis_ticks = np.concatenate([-half_axis_ticks, half_axis_ticks, [0]])

for i_N, N in enumerate(Ns):
    save_name = results_dir + fr'\plots\N={N}.png'

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in roots_dict[N]]
    roots_imag = [np.imag(r) for r in roots_dict[N]]

    plt.figure(f'N={N}')

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
    median_root = median_roots[i_N]
    circle_median = plt.Circle((0, 0), median_roots[i_N], color='r', fill=False, label=fr'$|z|={{{median_root:.3f}}}$')
    plt.gca().add_patch(circle_median)

    plt.xscale('symlog', linthresh=linthresh)
    plt.yscale('symlog', linthresh=linthresh)

    # Set axes
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)

    plt.xlabel(r'Re$(z)$')
    plt.ylabel(r'Im$(z)$')

    plt.legend(loc='upper right')

    plt.show()

    if save_fig:
        plt.savefig(DFUtils.create_filename(save_name))


plt.figure('min and max |root|')
plt.plot(Ns, min_roots, '.', label='min')
plt.plot(Ns, max_roots, '.', label='max')
#TODO: bootstrap estimate this
plt.plot(Ns, median_roots, '.', label='median')
plt.xlabel(r'$N$')
plt.yscale('log')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(results_dir + r'\plots\min_and_max_root.png'))