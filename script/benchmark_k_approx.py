import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import unitary_group
import datetime
import logging
import time
import itertools

from src.loop_hafnian_k_approx import loop_hafnian_approx_batch
from src.gbs_experiment import sduGBS
from src.utils import LogUtils, DFUtils


Ns = np.arange(4, 10, step=1)
w_label = 'w=1'
k_labels = ['1', '2', 'exact']
max_num_outputs = 100000

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
logging.info(
    f'Benchmark k-th order approximation by TVD for N={Ns} and {w_label}')



for N in Ns:

    if w_label == 'w=1':
        w = 1
    elif w_label[2] == 'N':
        exponent_w = float(w_label[4:])
        w = N ** exponent_w
    else:
        raise Exception('w_label not recognized')

    #  design experiment
    M = N ** 2
    K = N
    U = unitary_group.rvs(M)

    experiment = sduGBS(M)
    r, beta = experiment.create_d_gbs(K, N, w, U)
    B = experiment.calc_B()
    gamma = experiment.calc_half_gamma()

    if comb(M, N) <= max_num_outputs:

        num_outputs = comb(M, N)
        outputs = np.zeros( (num_outputs, N), dtype=int)
        lhaf_vals = {}
        for k_label in k_labels:
            lhaf_vals[k_label] = np.zeros(num_outputs, dtype = np.complex128)

        i = 0
        for output in itertools.combinations(M, N):

            output_ports = np.asarray(output, dtype=int)
            outputs[0] = output_ports

            B_n = B[output_ports][:, output_ports]
            gamma_n = gamma[output_ports]

            for k_label in k_labels:

                if k_label == 'exact':
                    k = N // 2 + 1
                else:
                    k = int(k_label)

                lhaf_value = loop_hafnian_approx_batch(B_n, gamma_n, k=k)
                lhaf_vals[k_label][i] = lhaf_value

            i += 1

    else:
        num_outputs = max_num_outputs


