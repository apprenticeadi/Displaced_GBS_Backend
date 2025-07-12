import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import datetime
import string
from scipy.stats import bootstrap
import pandas as pd

from src.utils import DFUtils
from src.photon_number_distributions import total_displaced_squeezed_vacuum, vac_prob_displaced_squeezed_vacuum


result_timestamps = {
    'K=M_r=12.5': r'2024-02-07(10-40-20.147904)',  # r is squeezing/displacement average photon ratio. r=12.5 means w=0.1, r=0.17 means w=1.
    'K=M_r=0.17': r'2024-02-06(10-52-26.733207)',
    'K=M_r=1': r'2024-02-05(11-48-09.576129)',
    'K=sqrtM_r=1': r'2024-02-04(20-59-12.470099)',
    'K=sqrtM_r=12.5': r'2024-01-12(20-45-20.487901)',
    'K=sqrtM_r=0.17': r'2024-01-10(20-59-32.955839)',
}

time_now = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plot_dir = fr'..\Plots\acc_over_n\{time_now}'
result_par_dir = r'..\Results\anticoncentration_over_n'

alphabet = list(string.ascii_lowercase)
M_to_plots = [9, 16, 25]
r = 12.5
K_to_plots = ['K=M', 'K=sqrtM']

fontsize=14

'''Combined plot'''
figc, axc = plt.subplot_mosaic('aabbcc;ddeeff;ggghhh', figsize=(15,10), layout='constrained')

'''Plot raw distributions of tilde pn'''
# fig, axs = plt.subplots(2, 3, squeeze=True, figsize=(8, 5), layout='constrained', sharex='col', sharey='all')
top_ax_list = list(axc.values())[:6]
axs = np.asarray(top_ax_list).reshape(2,3)

for i in range(axs.shape[0]):

    klabel = K_to_plots[i]
    if klabel == 'K=M':
        klatex = r'$K=M$'
    elif klabel == 'K=sqrtM':
        klatex = r'$K=\sqrt{M}$'
    else:
        raise ValueError(f'klabel {klabel} not recognized')

    timestamp = result_timestamps[rf'{klabel}_r={r}']

    for j in range(axs.shape[1]):
        ax = axs[i,j]

        M = M_to_plots[j]
        N = int(np.sqrt(M))
        num_samples = comb(M, N)

        result_dir = DFUtils.return_filename_from_head(result_par_dir + rf'\{timestamp}', rf'M={M}')

        norm_probs = np.load(result_dir + rf'\norm_probs.npy')
        repeat, num_prob = norm_probs.shape

        assert num_prob == num_samples  # I should have calculated all pn in Omega.

        for k in range(repeat):
            if num_prob < 1000:
                ax.plot(list(range(num_prob)), norm_probs[k, :], 'b.', alpha=0.1, markersize=1)
            else:
                ax.plot(list(range(num_prob)), norm_probs[k, :], color='blue', alpha=0.1, linewidth=0.1)

        ax.set_yscale('log')
        ax.set_ylim([1e-14, 1])

        ax.axhline(y= 1 / num_samples, xmin=0, xmax=num_prob, color='gray', ls='dashed', label=rf'$1/|\Omega|$')

        if j == 0:
            ax.set_title(rf'({alphabet[i*axs.shape[1]+j]}) $M={{{M}}}$, '+klatex, loc='left', fontsize=fontsize)
        else:
            ax.set_title(rf'({alphabet[i*axs.shape[1]+j]}) $M={{{M}}}$', loc='left', fontsize=fontsize)

        if j == 0:
            ax.set_ylabel(r'$\tilde{p}_{\mathbf{n}}$', fontsize=fontsize-2)
        if i == 1:
            ax.set_xlabel(r'$\mathbf{n}$',fontsize=fontsize-2)

        if i == 0 and j == 0:
            ax.legend(fontsize=fontsize)

# fig.savefig(DFUtils.create_filename(plot_dir + rf'\r={r}_pn_distribs.pdf'))


'''Plot alpha for eta=0.25'''
rs = [12.5,1,0.17]

# figg, axx = plt.subplots(1, 2, squeeze=True, figsize=(12, 5), sharey=True)

bottom_ax_list = list(axc.values())[6:]
axx = np.asarray(bottom_ax_list)

for i_k, klabel in enumerate(K_to_plots):
    if klabel == 'K=M':
        klatex = r'$K=M$'
    elif klabel == 'K=sqrtM':
        klatex = r'$K=\sqrt{M}$'
    else:
        raise ValueError(f'klabel {klabel} not recognized')

    ax = axx[i_k]

    for r in rs:

        timestamp = result_timestamps[rf'{klabel}_r={r}']

        results_df = pd.read_csv(result_par_dir + rf'\{timestamp}\alpha_for_eta=0.25.csv')
        # the alpha here is defined as Pr(pn > alpha/|Omega|) > 1-eta

        inverse_alphas = np.asarray(results_df['alpha'])
        n_errors = np.asarray(results_df['n_error'])
        p_errors = np.asarray(results_df['p_error'])
        ax.errorbar(results_df['M'], 1/inverse_alphas, yerr=[n_errors/inverse_alphas**2, p_errors/inverse_alphas**2],
                     marker='.', label=rf'$R={r}$')

    if i_k == 0:
        ax.legend(fontsize=fontsize)
        ax.set_ylabel(r'$\alpha$', fontsize=fontsize-2)

    ax.set_title(rf'({alphabet[i_k+6]}) '+klatex + r', $\eta=25\%$', fontsize=fontsize, loc='left')
    ax.set_xlabel(r'$M$', fontsize=fontsize-2)
    ax.tick_params(labelsize=fontsize-4)
    ax.set_ylim([0, 25])
# figg.savefig(DFUtils.create_filename(plot_dir + rf'\r={rs}_alphas_against_M.pdf'))

figc.savefig(DFUtils.create_filename(plot_dir + rf'\combined_plot.pdf'))