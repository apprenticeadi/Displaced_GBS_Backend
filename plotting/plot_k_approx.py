import matplotlib.pyplot as plt
import numpy as np

from src.utils import  DFUtils

Ns = np.arange(4, 20, step=1)
w_label='w=3.7N^0.25'
output_num = 100
ks = [0, 1, 2, 3]

save_fig=True

results_dir = fr'..\Results\benchmark_k_approx\{w_label}_{output_num}outputs'

mean_rel_errs = np.zeros((len(Ns), len(ks)), dtype=np.float64)
for i_N, N in enumerate(Ns):

    N_dir = DFUtils.return_filename_from_head(results_dir, f'N={N}')

    outputs = np.load(N_dir + r'\outputs.npy')
    exact_lhafs = np.load(N_dir + r'\k=exact.npy')

    approx_lhafs = np.zeros((len(exact_lhafs), len(ks)), dtype=np.complex128)
    for i_k, k in enumerate(ks):
        approx_lhafs[:, i_k] = np.load(N_dir + rf'\k={k}.npy')

    exact_probs = np.absolute(exact_lhafs) ** 2
    approx_probs = np.absolute(approx_lhafs) ** 2

    relative_prob_errs =  (np.absolute(approx_probs.T-exact_probs)/exact_probs).T

    # plt.figure(f'relative errors N={N}')
    # for i_k, k in enumerate(ks):
    #     plt.hist(relative_prob_errs[:, i_k], bins=1000, range=[0, 1], alpha=0.5, label=f'k={k}')
    #
    # plt.legend()
    # plt.title(f'Relative errors N={N}')

    mean_rel_errs[i_N, :] = np.mean(relative_prob_errs, axis=0)

plt.figure('mean relative error')
for i_k, k in enumerate(ks):
    plt.plot(Ns, mean_rel_errs[:, i_k], label=f'k={k}')
plt.legend()
plt.xlabel('N')
plt.ylabel('Mean relative error')
plt.ylim(0, 5)
plt.title(f'Accuracy of k-th order approximation for {w_label}')

if save_fig:
    plt.savefig(DFUtils.create_filename(results_dir + r'\Plots\Mean_relative_error.png'))