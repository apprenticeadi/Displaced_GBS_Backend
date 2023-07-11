import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.utils import DFUtils
import datetime

time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

# <<<<<<<<<<<<<<<<<<< Bounded probability scale plot  >>>>>>>>>>>>>>>>>>
result_dir = r'..\Results\sharpP_prob\11-07-2023(15-24-53.587074)'
# bounded_probs = np.load(result_dir + r'\probs.npy')
#
# Ns = np.linspace(10, 500, num=20, dtype=int)
# w_labels = ['w=0.01', 'w=0.1', 'w=0.2', 'w=0.5']
# plt.figure('Bounded probability')
# for i_w, w_label in enumerate(w_labels):
#     plt.plot(Ns, bounded_probs[:, i_w], 'x', label=w_label)
# plt.legend()

# <<<<<<<<<<<<<<<<<<< Colour plots >>>>>>>>>>>>>>>>>>
make_from_scratch = True

if make_from_scratch:
    num_points = 100
    N_sqs = np.linspace(start=0.01, stop=1., num=num_points)
    rs = np.arcsinh(np.sqrt(N_sqs))

    N_diss = np.linspace(start=0.01, stop=1., num=num_points)
    betas = np.sqrt(N_diss)

    w_grid = np.outer((1 - np.tanh(rs)) / np.sqrt(np.tanh(rs)), betas)  # Each row is same r, each column is same beta

Ns = [100, 500]

for N in Ns:

    K = N
    M = N

    if make_from_scratch:
        save_dir = result_dir + fr'\N={N}_K={K}_M={M}\probs_{time_stamp}'
        raw_dir = result_dir + fr'\N={N}_K={K}_M={M}\raw'
        raw_files = os.listdir(raw_dir)

        bound_grid = w_grid ** 2 / (4 * N - 8)
        probs = np.zeros((len(raw_files), num_points, num_points), dtype=np.float64)
        probs_allbounded = np.zeros((num_points, num_points), dtype=np.float64)

        for i_file, raw_file in enumerate(raw_files):
            tildeX = np.load(raw_dir + fr'\{raw_file}')  # an 1d array, diagonal terms already filtered out
            abs_tildeX = np.absolute(tildeX)

            t0 = time.time()
            mask = np.less_equal.outer(bound_grid, abs_tildeX).astype(int)
            probs[i_file,:,:] = np.sum(mask, axis=-1) / len(abs_tildeX)
            t1 = time.time()

            probs_allbounded += (probs[i_file, :, :] == 1.).astype(int)

            print(f'N={N}, {raw_file}, time={t1-t0}')

        probs_allbounded = probs_allbounded / len(raw_files)

        np.save(DFUtils.create_filename(save_dir + r'\rs.npy'), rs)
        np.save(save_dir + r'\betas.npy', betas)
        np.save(save_dir + r'\w_grid.npy', w_grid)
        np.save(save_dir + r'\probs.npy', probs)
        np.save(save_dir + r'\probs_allbounded.npy', probs_allbounded)

    else:
        save_dir = DFUtils.return_filename_from_head(result_dir + fr'\N={N}_K={K}_M={M}', 'probs')
        probs_allbounded = np.load(save_dir + r'\probs_allbounded.npy')
        probs = np.load(save_dir + r'\probs.npy')

        rs = np.load(save_dir + r'\rs.npy')
        betas = np.load(save_dir + r'\betas.npy')
        w_grid = np.load(save_dir + r'\w_grid.npy')

        N_sqs = np.sinh(rs) ** 2
        N_diss = betas ** 2

    plt.figure(f'color map N={N}')
    plt.pcolormesh(N_diss, N_sqs, probs_allbounded)
    plt.colorbar()
    plt.xlabel(r'$\beta^2$')
    plt.ylabel(r'$\sinh^2(r)$')

    plt.plot(N_diss, 1-N_diss, color='red', linestyle='-')
    plt.ylim([min(N_sqs), max(N_sqs)])
    plt.xlim([min(N_diss), max(N_diss)])
    plt.title(fr'Pr$($all $\tilde{{B}}_{{ij}}>\frac{{1}}{{4(N-2)}})$ for M={M},K={K},N={N}')

    # plt.show()

    plt.savefig(DFUtils.create_filename(save_dir + r'\cmap_probs_allbounded.png'))

    av_probs = np.sum(probs, axis=0) / probs.shape[0]
    plt.figure(f'color map probs N={N}')
    plt.pcolormesh(N_diss, N_sqs, av_probs)
    plt.colorbar()
    plt.xlabel(r'$\beta^2$')
    plt.ylabel(r'$\sinh^2(r)$')

    plt.plot(N_diss, 1 - N_diss, color='red', linestyle='-')
    plt.ylim([min(N_sqs), max(N_sqs)])
    plt.xlim([min(N_diss), max(N_diss)])
    plt.title(fr'M={M},K={K},N={N}')

    plt.savefig(DFUtils.create_filename(save_dir + r'\cmap_probs.png'))