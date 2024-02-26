import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import datetime
from scipy.stats import bootstrap
import pandas as pd

from src.utils import DFUtils
from src.photon_number_distributions import total_displaced_squeezed_vacuum, vac_prob_displaced_squeezed_vacuum


time_stamp = r'\2024-01-12(20-45-20.487901)'
time_now = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")

Ms = np.arange(9, 26)
dir = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\anticoncentration_over_n' + time_stamp
savefig = False
if savefig:
    plot_dir = dir + rf'\plots_{time_now}'
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Definition:  Pr(p_n > alpha/num_probs) > 1-eta. For eta=constant, ACC wants alpha > 1/poly(n)

eta = 0.25
alphas = np.zeros((len(Ms), 4), dtype=float)
def find_quant(a, axis=0):
    return np.quantile(a, 1 - eta, axis=axis)

for it_M, M in enumerate(Ms):
    N_mean = np.sqrt(M)
    N_int = int(np.floor(np.sqrt(M)))
    K = N_int

    norm_probs = np.load(dir + fr'\M={M}_N={N_int}_K={K}\norm_probs.npy')
    repeat, num_prob = norm_probs.shape

    print(f'bootstrapping for M={M}')
    t1 = time.time()
    unnorm_p = norm_probs.flatten() * num_prob
    alpha = find_quant(unnorm_p)
    bstrp_res = bootstrap((unnorm_p,), find_quant, confidence_level=0.95)
    alphas[it_M, :] = (M, alpha, alpha - bstrp_res.confidence_interval.low, bstrp_res.confidence_interval.high - alpha)
    t2 = time.time()
    print(f'boostrapping complete after {t2-t1}s')

    plt.figure(f'M={M}_N={N_int}')
    for i in range(repeat):
        if num_prob < 1000:
            plt.plot(list(range(num_prob)), norm_probs, 'b.', alpha=0.1, markersize=1)
        else:
            plt.plot(list(range(num_prob)), norm_probs, color='blue', alpha=0.1, linewidth=0.1)

    plt.ylabel(r'$p_{\mathbf{n}}$')
    plt.xlabel(r'$\mathbf{n}$')

    plt.yscale('log')
    plt.title(f'M={M}, N={N_int}, K={K}')
    plt.xticks([0, num_prob])
    plt.axhline(y= 1 / num_prob, xmin=0, xmax=num_prob, color = cycle[0], label=rf'$1/|\Omega|$')
    plt.axhline(y=alpha / num_prob, xmin=0, xmax=num_prob, color=cycle[0], label=rf'${{{alpha:.3f}}}/|\Omega|$')
    if savefig:
        plt.savefig(DFUtils.create_filename(plot_dir + fr'\plot_M={M}_N={N_int}_K={K}.pdf'))

alpha_df = pd.DataFrame(alphas, columns=['M', 'alpha', 'n_error', 'p_error'])
alpha_df.to_csv(dir + rf'\alpha_for_eta={eta}.csv', index=False)


plt.figure(f'alpha vs M')
plt.errorbar(alphas[:, 0], alphas[:, 1], yerr=np.array([alphas[:, 2], alphas[:, 3]]))
plt.xlabel(r'$M$')
plt.ylabel(r'$\alpha$')
plt.ylim([0, 1])
plt.legend()
plt.title(rf'$\alpha$ for $\eta$={eta}')
if savefig:
    plt.savefig(plot_dir + fr'\plot_alpha_for_eta={eta}.pdf')
