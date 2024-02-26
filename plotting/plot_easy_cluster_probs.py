import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.utils import DFUtils, DGBSUtils, LogUtils
import datetime
import logging

'''
Plot the applicability regimes for cluster truncation method
'''

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")

results_timestamp = r'01-06-2023(16-25-37.414272)'
result_dir = r'..\Results\sharpP_prob' + fr'\{results_timestamp}'

save_dir = r'..\Results\easy_cluster_prob' + fr'\{results_timestamp}'

LogUtils.log_config(time_stamp=time_stamp, dir=save_dir, filehead='log', module_name='', level=logging.INFO)

w_labels = ['w=5.444', 'w=9.900', 'w=14.887', 'w=17.727']
Ks = np.linspace(10, 500, num=20, dtype=int) # what we care about should be K

logging.info(f'Read the raw tildeX data from {results_timestamp}, whose time-stamp defines the timestamp of the save '
             f'directory of this plotting script. '
             f'\nFor each K from {Ks}, the number of tildeX matrices are num_trial. '
             f'Therefore, there are (num_trial * (N^2-N)) tildeX_ij elements. '
             f'\nFor each w from {w_labels}, we record the '
             f'percentage of tildeX_ij out of (num_trial * (N^2-N)) that are bounded by w^2/e(2N-3) for each K in '
             f'probs_w_label.npy; \nand the percentage of tildeXs out of (num_trial) which have all elements bounded by'
             f'w^2/e(2N-3) for each K in all_bound_probs_w_label.npy. The Ks are recorded in Ks.npy')

np.save(DFUtils.create_filename(save_dir + fr'\Ks_{time_stamp}.npy'), Ks)

probs = np.zeros((len(w_labels), len(Ks)))  # the probability of an element being bounded, over 1000 trials * (N^2-N) elements per trial
all_bound_probs = np.zeros((len(w_labels), len(Ks)))  # the probability of all elements being bounded, over 1000 trials

for i_K, K in enumerate(Ks):

    M = K
    N = K

    raw_dir = result_dir + fr'\N={N}_K={K}_M={M}\raw'
    raw_files = os.listdir(raw_dir)
    num_trials = len(raw_files)

    ws = [DGBSUtils.read_w_label(w_label, N) for w_label in w_labels]
    bounds = np.power(ws, 2) / (np.e * (2 * N - 3))

    prob = np.zeros(len(ws))
    all_bounded_prob = np.zeros(len(ws), dtype=int)
    for i_file, raw_file in enumerate(raw_files):

        t0 = time.time()
        tildeX = np.load(raw_dir + fr'\{raw_file}')  # an 1d array, diagonal terms already filtered out
        abs_tildeX = np.absolute(tildeX)

        mask = np.greater_equal.outer(bounds, abs_tildeX).astype(int)

        new_add_prob = np.sum(mask, axis = -1) / len(abs_tildeX)
        prob += new_add_prob
        all_bounded_prob += (new_add_prob==1.).astype(int)  # array of shape (len(bounds),)

        t1 = time.time()

        logging.info(f'N={N}, {raw_file}, time={t1-t0}, prob={prob}')

    prob = prob / num_trials
    all_bound_prob = all_bounded_prob / num_trials

    probs[:, i_K] = prob
    all_bound_probs[:, i_K] = all_bound_prob


for i_w in range(len(w_labels)):
    np.save(save_dir + fr'\probs_{w_labels[i_w]}_{time_stamp}.npy', probs[i_w, :])
    np.save(save_dir + fr'\all_bound_probs_{w_labels[i_w]}_{time_stamp}.npy', all_bound_probs[i_w, :])

    plt.figure('Probs')
    plt.plot(Ks, probs[i_w, :], label=w_labels[i_w])

    plt.figure('All bound probs')
    plt.plot(Ks, all_bound_probs[i_w, :], label=w_labels[i_w])

plt.figure('Probs')
plt.legend()
plt.ylim([0, 1])
plt.xticks(Ks[::3])
plt.savefig(save_dir + fr'\probs_{time_stamp}.png')

plt.figure('All bound probs')
plt.legend()
plt.ylim([0,1])
plt.xticks(Ks[::3])
plt.xlabel(r'$N=K=\sqrt{M}$')
plt.savefig(save_dir + fr'\all_bound_probs_{time_stamp}.png')