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


Ns = np.arange(10, 19, step=1)
ks = np.asarray([0, 1, 2, 3, 4, 5])
max_num_outputs = 1000

ratios = [9]  # displacement over squeezing ratio
mean_photon_per_mode = 1.

# plt.figure('ln runtime comparison')
# plt.plot(Ns * np.log(2), label='exact')
# for k in range(1, 6):
#     plt.plot(np.log(comb(Ns, 2*k)) + 2*k*np.log(2), label=rf'$k={{{k}}}$')
#
# plt.xlabel('N')
# plt.ylabel('ln(runtime)')
# plt.legend()

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
LogUtils.log_config(time_stamp=time_stamp, filehead='benchmark_k_approx', module_name='', level=logging.INFO)
logging.info(f'Benchmark k-th order approximation for N={Ns}, k={ks}, N_dis/N_sq ratio = {ratios}. '
             f'The maximum number of outputs for which the lhaf is calculated is capped by {max_num_outputs}. ')

t_initial = time.time()
for ratio in ratios:
    N_sq = mean_photon_per_mode / (ratio + 1)  # sinh^2(r)
    r = np.arcsinh(np.sqrt(N_sq))
    N_dis = ratio * N_sq
    beta = np.sqrt(N_dis)

    w = DGBSUtils.calc_w(r, beta)
    w_label = fr'w={w:.3f}'

    results_dir = rf'..\Results\benchmark_k_approx\{w_label}_{max_num_outputs}outputs'

    logging.info(f'N_dis/N_sq ratio={ratio}, w={w}, r={r}, beta={beta}')

    for i_N, N in enumerate(Ns):

        # w = DGBSUtils.read_w_label(w_label, N)

        # <<<<<<<<<<<<<<<<<<< Design Experiment  >>>>>>>>>>>>>>>>>>
        M = N ** 2
        K = N
        U = unitary_group.rvs(M)

        results_N_dir = results_dir + fr'\N={N}_M={M}_K={K}_{time_stamp}'

        experiment = sduGBS(M)

        # r, beta = experiment.create_d_gbs(K, N, w, U)

        rs = np.concatenate([r * np.ones(K), np.zeros(M-K)])
        betas = np.concatenate([beta * np.ones(K), np.zeros(M-K)])
        experiment.add_squeezing(rs)
        experiment.add_displacement(betas)
        experiment.add_interferometer(U)

        B = experiment.calc_B()
        gamma = experiment.calc_half_gamma()

        np.save(DFUtils.create_filename(results_N_dir + r'\B.npy'), B)
        np.save(results_N_dir + r'\gamma.npy', gamma)
        np.save(results_N_dir + r'\U.npy', U)

        num_outputs = np.min((int(comb(M, N)), max_num_outputs))
        outputs = np.zeros( (num_outputs, N), dtype=int)
        lhaf_exact = np.zeros(num_outputs, dtype=np.complex128)
        lhaf_approx = np.zeros((num_outputs, len(ks)), dtype=np.complex128)  # Each column corresponds to a k-value

        logging.info(f'Generate experiment for M={M}, K=N={N}. Calculate lHaf for {num_outputs} outputs.')

        # <<<<<<<<<<<<<<<<<<< Calculate lhafs  >>>>>>>>>>>>>>>>>>
        for i in range(num_outputs):
            t0 = time.time()
            output_ports = np.asarray(random.sample(range(M), N), dtype=int)

            # let's ignore duplicates for now. For some reason the program gets stuck when I include the following part.
            # while output_ports in outputs[:i]:
            #     output_ports = np.asarray(random.sample(range(M), N), dtype=int)

            outputs[i] = output_ports
            t1 = time.time()

            B_n = B[output_ports][:, output_ports]
            gamma_n = gamma[output_ports]

            lhaf_exact_val = loop_hafnian(B_n, gamma_n)
            lhaf_exact[i] = lhaf_exact_val

            for i_k, k in enumerate(ks):
                if k >= N // 2:
                    lhaf_value = lhaf_exact_val
                else:
                    lhaf_value = loop_hafnian_approx_batch(B_n, gamma_n, k=k)

                lhaf_approx[i, i_k] = lhaf_value

            t2 = time.time()
            logging.info(f'{i}-th output={output_ports}, calc loop Hafnians time ={t2-t1}')

        # <<<<<<<<<<<<<<<<<<< Save lhaf results  >>>>>>>>>>>>>>>>>>
        np.save(results_N_dir + '\outputs.npy', outputs)
        np.save(results_N_dir + '\k=exact.npy', lhaf_exact)
        for i_k, k in enumerate(ks):
            np.save(results_N_dir + f'\k={k}.npy', lhaf_approx[:, i_k])

t_final = time.time()

print(f'Total time = {t_final - t_initial}')