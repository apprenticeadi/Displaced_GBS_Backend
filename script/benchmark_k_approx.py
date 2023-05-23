import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import unitary_group
import datetime
import logging
import time
import itertools
import random

from thewalrus import loop_hafnian

from src.loop_hafnian_k_approx import loop_hafnian_approx_batch
from src.gbs_experiment import sduGBS
from src.utils import LogUtils, DFUtils, DGBSUtils


Ns = np.arange(4, 20, step=1)
w_label = 'w=0.09N^0.25'
ks = np.asarray([0, 1, 2])
max_num_outputs = 10000


# plt.figure('runtime comparison')
# plt.plot(np.power(2, Ns), label=r'$2^N$')
# for k in range(1, 3):
#     plt.plot(comb(Ns, 2*k) * np.power(2, 2*k), label=rf'$k={{{k}}}$')
#
# plt.xlabel('N')
# plt.ylabel('runtime')
# plt.legend()
# plt.yscale('log')

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
LogUtils.log_config(time_stamp=time_stamp, filehead='benchmark_k_approx', module_name='', level=logging.INFO)
logging.info(f'Benchmark k-th order approximation for N={Ns}, {w_label} and k={ks}. The maximum number of outputs '
             f'for which the lhaf is calculated is capped by {max_num_outputs}. ')

results_dir = rf'..\Results\benchmark_k_approx\{w_label}_{max_num_outputs}outputs'

for i_N, N in enumerate(Ns):

    w = DGBSUtils.read_w_label(w_label, N)

    # <<<<<<<<<<<<<<<<<<< Design Experiment  >>>>>>>>>>>>>>>>>>
    M = N ** 2
    K = N
    U = unitary_group.rvs(M)

    results_N_dir = results_dir + fr'\N={N}_M={M}_K={K}_{time_stamp}'

    experiment = sduGBS(M)
    r, beta = experiment.create_d_gbs(K, N, w, U)
    B = experiment.calc_B()
    gamma = experiment.calc_half_gamma()

    np.save(DFUtils.create_filename(results_N_dir + r'\B.npy'), B)
    np.save(results_N_dir + r'\gamma.npy', gamma)
    np.save(results_N_dir + r'\U.npy', U)

    num_outputs = np.min((int(comb(M, N)), max_num_outputs))
    outputs = np.zeros( (num_outputs, N), dtype=int)
    lhaf_exact = np.zeros(num_outputs, dtype=np.complex128)
    lhaf_approx = np.zeros((num_outputs, len(ks)), dtype=np.complex128)  # Each column corresponds to a k-value

    logging.info(f'Generate experiment for M={M}, K=N={N}, r={r}, beta={beta}. '
                 f'Calculate lHaf for {num_outputs} outputs.')

    # <<<<<<<<<<<<<<<<<<< Calculate lhafs  >>>>>>>>>>>>>>>>>>
    for i in range(num_outputs):
        output_ports = np.asarray(random.sample(range(M), N), dtype=int)
        while output_ports in outputs:
            output_ports = np.asarray(random.sample(range(M), N), dtype=int)
        outputs[i] = output_ports

        B_n = B[output_ports][:, output_ports]
        gamma_n = gamma[output_ports]

        t1 = time.time()
        lhaf_exact_val = loop_hafnian(B_n, gamma_n)
        lhaf_exact[i] = lhaf_exact_val

        for i_k, k in enumerate(ks):
            if k >= N // 2:
                lhaf_value = lhaf_exact_val
            else:
                lhaf_value = loop_hafnian_approx_batch(B_n, gamma_n, k=k)

            lhaf_approx[i, i_k] = lhaf_value

        t2 = time.time()
        logging.info(f'{i}-th output={output_ports}, time={t2-t1}, exact |lhaf|^2={np.absolute(lhaf_exact_val)**2}')

    # <<<<<<<<<<<<<<<<<<< Save lhaf results  >>>>>>>>>>>>>>>>>>
    np.save(results_N_dir + '\outputs.npy', outputs)
    np.save(results_N_dir + '\k=exact.npy', lhaf_exact)
    for i_k, k in enumerate(ks):
        np.save(results_N_dir + f'\k={k_label}.npy', lhaf_approx[:, i_k])
