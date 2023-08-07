import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.utils import DFUtils
import datetime

time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")


result_dir = r'..\Results\sharpP_prob\11-07-2023(15-01-45.332916)'
make_from_scratch = False
if make_from_scratch:
    num_points = 100  # How many values we take between 0 and 1 for N_diss and N_sqs
    N_sqs = np.linspace(start=0.01, stop=1., num=num_points)
    rs = np.arcsinh(np.sqrt(N_sqs))

    N_diss = np.linspace(start=0.01, stop=1., num=num_points)
    betas = np.sqrt(N_diss)
else:
    read_folder_id = 1  # which 'probs' folder to read from

# <<<<<<<<<<<<<<<<<<< Bounded probability scale plot  >>>>>>>>>>>>>>>>>>
# bounded_probs = np.load(result_dir + r'\probs.npy')
#
# Ns = np.linspace(10, 500, num=20, dtype=int)
# w_labels = ['w=0.01', 'w=0.1', 'w=0.2', 'w=0.5']
# plt.figure('Bounded probability')
# for i_w, w_label in enumerate(w_labels):
#     plt.plot(Ns, bounded_probs[:, i_w], 'x', label=w_label)
# plt.legend()

# <<<<<<<<<<<<<<<<<<< Colour plots >>>>>>>>>>>>>>>>>>
Ks = np.array([10, 15, 20])  # what we care about should be K

for K in Ks:

    M = K**2
    N = K


    if make_from_scratch:

        raw_dir = result_dir + fr'\N={N}_K={K}_M={M}\raw'
        raw_files = os.listdir(raw_dir)

        bound_grid = np.zeros((num_points, num_points))

        for i_r, r in enumerate(rs):
            for i_beta, beta in enumerate(betas):
                w = beta * (1-np.tanh(r)) / np.sqrt(np.tanh(r))
                av_N = max(int(K * (np.sinh(r)**2 + beta**2) + 0.5), 3)  # closest integer or 3

                bound_grid[i_r, i_beta] = w**2 / (4*av_N - 8)

        save_dir = result_dir + fr'\N={N}_K={K}_M={M}\probs_{time_stamp}'

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
        np.save(save_dir + r'\bound_grid.npy', bound_grid)
        np.save(save_dir + r'\probs.npy', probs)
        np.save(save_dir + r'\probs_allbounded.npy', probs_allbounded)

    else:
        save_dir = DFUtils.return_filename_from_head(result_dir + fr'\N={N}_K={K}_M={M}', 'probs', idx=read_folder_id)
        probs_allbounded = np.load(save_dir + r'\probs_allbounded.npy')
        probs = np.load(save_dir + r'\probs.npy')

        rs = np.load(save_dir + r'\rs.npy')
        betas = np.load(save_dir + r'\betas.npy')

        N_sqs = np.sinh(rs) ** 2
        N_diss = betas ** 2

    # Filter out the data where the total average photon number is few than 3
    #TODO: fix the value of k in np.tri
    mask2 = np.tri(probs_allbounded.shape[0], k= int(3 / K / 0.01) - 100, dtype=bool)
    masked_probs_allbounded = np.ma.masked_where(mask2, probs_allbounded[::-1, :])
    masked_probs_allbounded = masked_probs_allbounded[::-1, :]

    plt.figure(f'color map K={N}')
    plt.pcolormesh(N_diss, N_sqs, masked_probs_allbounded)
    plt.colorbar()
    plt.xlabel(r'$\beta^2$')
    plt.ylabel(r'$\sinh^2(r)$')

    plt.plot(N_diss, 1-N_diss, color='red', linestyle='-')
    plt.ylim([min(N_sqs), max(N_sqs)])
    plt.xlim([min(N_diss), max(N_diss)])
    # plt.title(fr'Pr$($all $\tilde{{B}}_{{ij}}>\frac{{1}}{{4(N-2)}})$ for M={M},K={K},N={N}')

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
    # plt.title(fr'M={M},K={K},N={N}')

    plt.savefig(DFUtils.create_filename(save_dir + r'\cmap_probs.png'))