import matplotlib.pyplot as plt
import numpy as np
from src.utils import DFUtils
import datetime

total_Ns = list(range(4, 17))

Ns_dir_dict = {
    r'..\Results\roots_matching_polynomial\10000rep_2023-08-23(18-00-43.689530)': list(range(4, 15)),
    r'..\Results\roots_matching_polynomial\10000rep_2023-08-23(19-35-17.526866)': [15],
    r'..\Results\roots_matching_polynomial\10000rep_2023-08-25(17-08-36.941174)': [16]
}

save_fig = True
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plot_dir = fr'..\Plots\roots_mp\tildeX_distrib_{time_stamp}'

roots_dict = {}
min_roots_mp_dict={}
min_roots = np.zeros(len(total_Ns), dtype=float)
max_roots = np.zeros(len(total_Ns), dtype=float)
median_roots = np.zeros(len(total_Ns), dtype=float)
median_min_roots_mp = np.zeros(len(total_Ns), dtype=float)
for results_dir in Ns_dir_dict.keys():
    Ns = Ns_dir_dict[results_dir]
    for N in Ns:
        roots_N = np.load(results_dir + fr'\N={N}_roots.npy')
        abs_roots_N = np.abs(roots_N)

        roots_dict[N] = roots_N  # this is all the roots

        i_N = total_Ns.index(N)

        min_roots[i_N] = np.min(abs_roots_N)
        max_roots[i_N] = np.max(abs_roots_N)
        median_roots[i_N] = np.median(abs_roots_N)

        # the min root from each distinct matching polynomial
        roots_N_mp = roots_N.reshape((10000, N//2))
        row_args = np.abs(roots_N_mp).argmin(axis=1)  # this probably is always the last element
        min_roots_mp = np.zeros(10000, dtype=np.complex128)
        for i_row, row_arg in enumerate(row_args):
            min_roots_mp[i_row] = roots_N_mp[i_row, row_arg]
        min_roots_mp_dict[N] = min_roots_mp
        median_min_roots_mp[i_N] = np.median(np.abs(min_roots_mp))

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
log_linthresh = int(np.log10(np.min(min_roots))) - 1
linthresh = 10 ** (log_linthresh)
axis_log_lim = int(np.log10(np.max(max_roots))) + 1
axis_lim = 10 ** (axis_log_lim)
half_axis_ticks = 10 ** np.arange(log_linthresh, axis_log_lim+1, step=2, dtype=np.float64)
axis_ticks = np.concatenate([-half_axis_ticks, half_axis_ticks, [0]])


for i_N, N in enumerate(total_Ns):
    save_name = plot_dir + fr'\N={N}.png'

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


# plot the min root of each matching polynomial
for i_N, N in enumerate(total_Ns):
    save_name = plot_dir + fr'\min_root_of_each_mp_N={N}.png'

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in min_roots_mp_dict[N]]
    roots_imag = [np.imag(r) for r in min_roots_mp_dict[N]]

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
    median_root = median_min_roots_mp[i_N]
    circle_median = plt.Circle((0, 0), median_root, color='r', fill=False, label=fr'$|z|={{{median_root:.3f}}}$')
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
plt.plot(total_Ns, min_roots, '.', label='min')
# plt.plot(total_Ns, max_roots, '.', label='max')
#TODO: bootstrap estimate this
plt.plot(total_Ns, median_roots, '.', label='median')
plt.plot(total_Ns, median_min_roots_mp, '.', label='median(min)')
plt.xlabel(r'$N$')
plt.yscale('log')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\min_and_max_root.png'))

# Colour plot
w_max = 5.000
w_min = 0.0001
ws = np.linspace(w_min, w_max, 10000)
epsilons = 1 / ws**2
epsilons = epsilons[::-1]

abs_min_roots_mp_arr = np.zeros((len(total_Ns), 10000), dtype=float)  # each row is the 10,000 abs(min(roots)) for each N
for i_N, N in enumerate(total_Ns):
    abs_min_roots_mp_arr[i_N] = np.abs(min_roots_mp_dict[N])

color_mesh = np.zeros((len(total_Ns), len(epsilons)), dtype=float)
for i_eps, epsilon in enumerate(epsilons):
    counts = np.sum(abs_min_roots_mp_arr <= epsilon, axis=1)
    color_mesh[:, i_eps] = counts / 10000

plt.figure('color plot Pr(hit a zero)')
plt.pcolormesh(np.concatenate([[2*ws[-1]-ws[-2]], ws[::-1]]), np.arange(min(total_Ns), max(total_Ns)+2)-0.5, color_mesh, vmin=0, vmax=1)
plt.colorbar()
plt.xlabel(r'$|w|$')
plt.ylabel(r'$N$')
plt.xscale('linear')
plt.ylim([min(total_Ns)-0.5, max(total_Ns)+0.5])
plt.yticks(total_Ns[::2])
plt.xlim([min(ws), max(ws)])
for N in total_Ns:
    plt.axhline(N-0.5, xmin=0, xmax=w_max, color='white', linestyle=':')

#TODO: bootstrap these
plt.plot(1 / np.sqrt(median_min_roots_mp), total_Ns, 'x', color='red')  # this is 50%

arg_quarter = np.argmax(color_mesh>=0.25, axis=1)
ws_reverse = ws[::-1]
plt.plot(ws_reverse[arg_quarter], total_Ns, 'x', color='black')  # this is 25%

plt.savefig(DFUtils.create_filename(plot_dir + r'\color_mesh_prob_hit_a_zero.png'))