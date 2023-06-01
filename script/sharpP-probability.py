import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
import numba
import time
import logging
import datetime

from src.utils import DGBSUtils, LogUtils, DFUtils

Ns = np.linspace(10, 500, num=20, dtype=int)
Ms = Ns
w_labels = ['w=0.01', 'w=0.1', 'w=0.2', 'w=0.5']

num_trials = 1000

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\sharpP_prob\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info(f'In this script, for each N of {Ns}, corresponding M are {Ms},'
             f'{num_trials} Haar random unitaries are generated, and for each '
             f'unitary U, tildeX = B_ij/gamma_i * gamma_j is calculated. Then for different w values out of {w_labels},'
             f'tildeX / w^2 is calculated, and the percentage of elements that are lower bounded by 1/(4(N-2)) is '
             f'calculated. After computing this for {num_trials} Haar random unitaries, the empirical probability '
             f'of having all tildeX/w^2 elements lower bounded is computed. ')

logging.info(f'w values are {w_labels}')

probs = np.zeros((len(Ns), len(w_labels)), dtype=np.float64)

for i_N, N in enumerate(Ns):
    M = Ms[i_N]
    K = N
    lower_bound = 1 / (4 * (N - 2))

    dir_N = dir + fr'\N={N}_K={K}_M={M}'

    bounded_percentages = np.zeros((num_trials, len(w_labels)), dtype=np.float64)
    for i in range(num_trials):
        t1 = time.time()
        U = unitary_group.rvs(M)  # this doesn't work with numba
        t2 = time.time()

        id_K = np.zeros(M)
        id_K[:K] = 1.
        id_K = np.diag(id_K)

        B = U @ id_K @ U.T  # The tanhr term is absorbed inside w
        half_gamma = np.sum(U[:, :K], axis=1)

        tildeX = B / np.outer(half_gamma, half_gamma)
        maskedX = tildeX[~np.eye(len(tildeX), dtype=bool)]  # mask out diagonal terms which are zero. masked shaped is 1d

        np.save(DFUtils.create_filename(dir_N + fr'\raw\tilde_X_{i}.npy'), maskedX)

        percentage = np.zeros(len(w_labels), dtype=np.float64)
        for i_w, w_label in enumerate(w_labels):
            w = DGBSUtils.read_w_label(w_label, N)

            percentage[i_w] = np.sum(np.absolute(maskedX / w ** 2) > lower_bound) / len(maskedX)

        bounded_percentages[i, :] = percentage
        logging.info(f'N={N}, {i}-th trial, time for U={t2-t1}, {percentage}')

    np.save(DFUtils.create_filename(dir_N + fr'\bounded_percentages.npy'), bounded_percentages)

    probs[i_N, :] = np.sum(bounded_percentages == 1., axis=0) / num_trials

np.save(DFUtils.create_filename(dir + fr'\probs.npy'), probs)

plt.figure('Bounded probability')
for i_w, w_label in enumerate(w_labels):
    plt.plot(Ns, probs[:, i_w], 'x', label=w_label)
plt.legend()
