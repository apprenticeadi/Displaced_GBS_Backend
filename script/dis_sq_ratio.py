import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.utils import DGBSUtils, DFUtils


# Ns = np.logspace(1, 5, num=101, dtype=int)
Ns = np.arange(4, 20, step=1)
w_label = 'w=N^0.25'

plot_dir = fr'..\Plots\dis_sq_ratio\{w_label}'
save_fig = False

results_dict = {
    'N': [], 'w': [], 'r': [], 'beta': [], 'ratio': []
}

for N in Ns:

    w = DGBSUtils.read_w_label(w_label, N)

    r, beta = DGBSUtils.solve_w(w, N_mean=1)
    w_calc = beta * (1 - np.tanh(r)) / np.sqrt(np.tanh(r))

    if np.isclose(w, w_calc):
        # print(f'True, N={N}, w={w}, w_calc={w_calc}, error={w_calc - w}')
        results_dict['N'].append(N)
        results_dict['w'].append(w)
        results_dict['r'].append(r)
        results_dict['beta'].append(beta)
        results_dict['ratio'].append(np.absolute(beta)**2 / np.sinh(r)**2)
    else:
        print(f'False, N={N}, w={w}, w_calc={w_calc}, error={w_calc - w}')

Ns_plot = np.asarray(results_dict['N'])
ws = np.asarray(results_dict['w'])
rs = np.asarray(results_dict['r'])
betas = np.asarray(results_dict['beta'])
ratios = np.asarray(results_dict['ratio'])

plt.figure('ratio')
plt.plot(Ns_plot, ratios, 'x')
# ratio_fit = stats.linregress(np.log(Ns_plot), np.log(ratios))
# plt.plot(Ns_plot,  np.exp(ratio_fit.intercept) * Ns_plot ** ratio_fit.slope, '--', color='black')
# plt.text(Ns_plot[-1] / 100, ratios[-1]/2, fr'$|\beta|^2/\sinh^2(r)={np.exp(ratio_fit.intercept):.3}*N^{{{ratio_fit.slope:.3}}}$ ')

# plt.plot(Ns_plot, Ns_plot-1, '-.', color='black')
plt.yscale('linear')
plt.xscale('linear')
plt.title(r'$|\beta|^2/\sinh^2(r)$ for ' + w_label)
plt.xlabel('N')
plt.ylabel('ratio')
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\ratio.png'))


plt.figure('total squeezing and displacement')
total_dis = Ns_plot * np.absolute(betas)**2
total_sq = Ns_plot * np.sinh(rs)**2

plt.plot(Ns_plot, total_sq, 'x', label='from squeezing')
plt.plot(Ns_plot, total_dis, 'x', label='from displacement')

dis_fit = stats.linregress(np.log(Ns_plot), np.log(total_dis))
plt.plot(Ns_plot, np.exp(dis_fit.intercept) * Ns_plot ** dis_fit.slope, '--', color='black')
plt.text(Ns_plot[-1] / 100, total_dis[-1]/2, fr'$|\beta|^2={np.exp(dis_fit.intercept):.3}*N^{{{dis_fit.slope:.3}}}$ ')

plt.title('Total mean photon number for ' + w_label)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Mean photon number')
if save_fig:
    plt.savefig(DFUtils.create_filename(plot_dir + r'\total_mean_photon_number.png'))
