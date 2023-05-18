import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import unitary_group
import datetime
import logging
import time
import itertools
import pandas as pd

from thewalrus import loop_hafnian

from src.loop_hafnian_k_approx import loop_hafnian_approx_batch
from src.gbs_experiment import sduGBS
from src.utils import LogUtils, DFUtils


Ns = np.arange(4, 20, step=1)
w_label = 'w=1'
k_labels = np.asarray(['0', '1', '2'])
max_num_outputs = 10000

tvd_dict = {}
tvd_dict['N'] = Ns
for k_label in k_labels:
    tvd_dict[k_label] = np.zeros(len(Ns), dtype=np.float64)

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
logging.info(f'Benchmark k-th order approximation by TVD for N={Ns}, {w_label} and k={k_labels}. The maximum number of outputs '
             f'for which the lhaf is calculated is capped by {max_num_outputs}. ')

results_dir = rf'..\Results\benchmark_k_approx\{w_label}_{max_num_outputs}outputs'

for i_N, N in enumerate(Ns):

    if w_label == 'w=1':
        w = 1
    elif w_label[2] == 'N':
        exponent_w = float(w_label[4:])
        w = N ** exponent_w
    else:
        raise Exception('w_label not recognized')

    # <<<<<<<<<<<<<<<<<<< Design Experiment  >>>>>>>>>>>>>>>>>>
    M = N ** 2
    K = N
    U = unitary_group.rvs(M)

    results_N_dir = results_dir + fr'\N={N}_M={M}_K={K}_{time_stamp}'

    experiment = sduGBS(M)
    r, beta = experiment.create_d_gbs(K, N, w, U)
    B = experiment.calc_B()
    gamma = experiment.calc_half_gamma()

    np.save(DFUtils.create_filename(results_N_dir + '\B.npy'), B)
    np.save(results_N_dir + '\gamma.npy', gamma)

    num_outputs = np.min((int(comb(M, N)), max_num_outputs))
    outputs = np.zeros( (num_outputs, N), dtype=int)
    lhaf_exact = np.zeros(num_outputs, dtype=np.complex128)
    lhaf_approx = np.zeros((num_outputs, len(k_labels)), dtype=np.complex128)  # Each column corresponds to a k-value

    logging.info(f'Generate experiment for M={M}, K=N={N}, r={r}, beta={beta}. '
                 f'Calculate lHaf for {num_outputs} outputs.')

    # <<<<<<<<<<<<<<<<<<< Calculate lhafs  >>>>>>>>>>>>>>>>>>
    i = 0
    for output in itertools.combinations(range(M), N):
        # If comb(M,N)> max_num_output, then I will just take the first [max_num_output] patterns of combinations(M,N),
        # instead of selecting random ones.
        if i >= num_outputs:
            break

        output_ports = np.asarray(output, dtype=int)
        outputs[i] = output_ports

        B_n = B[output_ports][:, output_ports]
        gamma_n = gamma[output_ports]

        t1 = time.time()
        lhaf_exact_val = loop_hafnian(B_n, gamma_n)
        lhaf_exact[i] = lhaf_exact_val

        for i_k, k_label in enumerate(k_labels):

            if k_label == 'exact':
                k = N // 2 + 1
                lhaf_value = loop_hafnian_approx_batch(B_n, gamma_n, k=k)
            k = int(k_label)


            if k >= N // 2:
                lhaf_value = lhaf_exact_val
            else:
                lhaf_value = loop_hafnian_approx_batch(B_n, gamma_n, k=k)


            lhaf_approx[i, i_k] = lhaf_value

        t2 = time.time()
        logging.info(f'{i}-th output={output_ports}, time={t2-t1}, exact |lhaf|^2={np.absolute(lhaf_exact_val)**2}')

        i += 1

    # <<<<<<<<<<<<<<<<<<< Save lhaf results  >>>>>>>>>>>>>>>>>>
    np.save(results_N_dir + '\outputs.npy', outputs)
    np.save(results_N_dir + '\k=exact.npy', lhaf_exact)
    for i_k, k_label in enumerate(k_labels):
        np.save(results_N_dir + f'\k={k_label}.npy', lhaf_approx[:, i_k])

    # # <<<<<<<<<<<<<<<<<<< Calculate TVD for each N  >>>>>>>>>>>>>>>>>>
    # lhaf_squared_exact = np.absolute(lhaf_exact) ** 2
    # lhaf_squared_approx = np.absolute(lhaf_approx) ** 2
    #
    # for i_k, k_label in enumerate(k_labels):
    #     tvd = 0.5 * np.sum(np.absolute(lhaf_squared_approx[:, i_k] - lhaf_squared_exact))
    #     logging.info(f'k={k_label}, tvd={tvd}')
    #     tvd_dict[k_label][i_N] = tvd

# TVD is not a good metric here
# # <<<<<<<<<<<<<<<<<<< Save TVD dataframe  >>>>>>>>>>>>>>>>>>
# tvd_df = pd.DataFrame(data=tvd_dict)
# tvd_df.to_csv(results_dir + f'\TVD_{time_stamp}.csv')
#
# plt.figure('TVD scaled')
# for k_label in k_labels:
#     plt.plot(tvd_dict['N'], comb(Ns**2, Ns) / 10 * tvd_dict[k_label], label=f'k={k_label}')
# plt.legend()
# plt.yscale('log')

