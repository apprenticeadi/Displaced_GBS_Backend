import numpy as np
import matplotlib.pyplot as plt
from src.utils import DFUtils
import math


time_stamp = r'\13-02-2023(19-04-51.793796)'

Ms = np.arange(9, 21)
dir = r'..\Results\anticoncentration' + time_stamp
savefig = False
if savefig:
    plot_dir = dir + r'\plots'
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

acc_beta_means = np.load(dir + r'\acc_beta_means.npy')
acc_beta_errs = np.load(dir + r'\acc_beta_errs.npy')

betas_dict = {}
# betas_dict[0.01] = np.load(dir + r'\acc_beta_means.npy'), np.load(dir + r'\acc_beta_errs.npy')
alphas = [1, 0.5, 0.1, 0.05, 0.01]
for alpha in alphas:
    betas_dict[alpha] = (np.zeros_like(Ms, dtype=float), np.zeros_like(Ms, dtype=float))


# TODO: work out betas for different alphas and plot them out.
for it_M, M in enumerate(Ms):
    N_int = int(np.floor(np.sqrt(M)))

    norm_probs = np.load(dir + fr'\M={M}_N={N_int}\norm_probs.npy')
    repeat, num_prob = norm_probs.shape

    betas_i = np.zeros((len(alphas), repeat), dtype=float)

    plt.figure(f'M={M}_N={N_int}')
    for i in range(repeat):

        for j, alpha in enumerate(alphas):
            betas_i[j, i] = np.count_nonzero(norm_probs[i] > alpha / num_prob) / num_prob

        plt.plot(list(range(num_prob)), norm_probs[i], ',')

    plt.yscale('log')
    plt.title('M={}, N={}'.format(M, N_int))
    plt.xticks([0, num_prob])
    for j, alpha in enumerate(alphas):
        betas_dict[alpha][0][it_M] = np.mean(betas_i[j, :])
        betas_dict[alpha][1][it_M] = np.std(betas_i[j, :])
        plt.axhline(y= alpha / num_prob, xmin=0, xmax=num_prob, color = cycle[j], label=f'{alpha}/({M} choose {N_int})')
    plt.legend()
    if savefig:
        plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_M={M}_N={N_int}.png'))

test = np.allclose(betas_dict[0.01][0], np.load(dir + r'\acc_beta_means.npy'))
test2 = np.allclose(betas_dict[0.01][1], np.load(dir + r'\acc_beta_errs.npy'))
print(test)
print(test2)



plt.figure(f'acc_betas')
for j, alpha in enumerate(alphas):
    plt.errorbar(Ms, betas_dict[alpha][0], yerr=betas_dict[alpha][1], color = cycle[j],label=f'alpha={alpha}')
plt.xlabel('M')
plt.ylabel('beta')
plt.ylim([0, 1])
plt.legend()
plt.title('beta against M')
if savefig:
    plt.savefig(plot_dir + fr'\plot_acc_betas.png')