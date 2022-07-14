import numpy as np
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils, DFUtils
from src.random_graph import RandomGraph
from src.gbs_matrices import GraphMatrices, GaussianMatrices
from src.log_utils import LogUtils

# <<<<<<<<<<<<<<<<<<< Parameters  >>>>>>>>>>>>>>>>>>
x = 1  # edge activity
r_max = 1  # maximum squeezing available in experiment
gamma_low = 1
gamma_high = np.sqrt(10)
M_list = [5, 8, 10, 13, 15, 18, 20]
log_diag_weights = list(range(-5, 6))  # log(w)

if x < 0:
    complexity = 'hard'
else:
    complexity = 'easy'
# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_even_v\x={}_{}'.format(x, time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('edge activity x={}, r_max={}, gamma_low={}, gamma_high={}, M_list={}, log_diag_weights={}'
             .format(x, r_max, gamma_low, gamma_high, M_list, log_diag_weights))


# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
# sq photons
fig1, ax1 = plt.subplots()
ax1.set_title('Average squeezed photons')
ax1.set_xlabel('log(w)')
ax1.set_ylabel('n_sq')
color_cycler = fig1.gca()._get_lines.prop_cycler

# cl photons
fig2, ax2 = plt.subplots()
ax2.set_title('Average classical photons')
ax2.set_xlabel('log(w)')
ax2.set_ylabel('n_cl')
ax2.set_yscale('log')

# sq/cl
fig3, ax3 = plt.subplots()
ax3.set_title('Photon number ratio squeezed/classical')
ax3.set_xlabel('log(w)')
ax3.set_ylabel('n_sq/n_cl')
ax3.set_yscale('log')

for M in M_list:
    # <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>
    delta = max(3, int(0.5 * M))  # maximum degree
    graph_G = RandomGraph(M=M, max_degree=delta)

    logging.info('')
    logging.info('M={} and max degree={}'.format(M, delta))
    logging.info('Graph adjacency matrix is \n{}'.format(graph_G.adjacency_matrix()))

    # <<<<<<<<<<<<<<<<<<< half gamma vector  >>>>>>>>>>>>>>>>>>
    half_gamma = np.random.uniform(low=gamma_low, high=gamma_high, size=M)
    Bmat = graph_G.generate_Bmatrix(x, half_gamma)

    logging.info('half_gamma={}'.format(half_gamma))

    # <<<<<<<<<<<<<<<<<<< Data file  >>>>>>>>>>>>>>>>>>
    save_filename = DFUtils.create_filename(
        dir + r'\M={}_delta={}'.format(M, delta))
    sq_phots = []
    coh_phots = []

    for idx, log_diag_weight in enumerate(log_diag_weights):

        # <<<<<<<<<<<<<<<<<<< B diagonal >>>>>>>>>>>>>>>>>>
        v = np.ones(M) * 10 ** log_diag_weight  # np.random.uniform(low=0, high=1, size=M) * 100000
        Bmat = MatrixUtils.filldiag(Bmat, v)
        logging.info('For diag_weight={}, v = {}'.format(10 ** log_diag_weight, v))

        # <<<<<<<<<<<<<<<<<<< rescale cB and Gamma >>>>>>>>>>>>>>>>>>
        eigs_B = np.linalg.eigvalsh(Bmat)
        c_factor = np.tanh(r_max) / abs(max(eigs_B, key=abs))
        cB = c_factor * Bmat
        Gamma = np.sqrt(c_factor) * np.concatenate([half_gamma, half_gamma.conjugate()])

        # <<<<<<<<<<<<<<<<<<< experimental parameters >>>>>>>>>>>>>>>>>>
        Amat = GraphMatrices.pure_A_from_B(cB)
        mu_fock = GaussianMatrices.mu_fock_from_A(Amat, Gamma)
        tanhr, U = takagi(cB)

        sq = np.arctanh(tanhr)
        displacement = mu_fock[:M]
        logging.info('sq = {}, displacement = {}'.format(sq, displacement))

        sq_phot = np.sum(np.sinh(sq) ** 2)
        coh_phot = np.sum(displacement * displacement.conjugate()).real

        sq_phots.append(sq_phot)
        coh_phots.append(coh_phot)

    sq_phots = np.array(sq_phots)
    coh_phots = np.array(coh_phots)

    df = pd.DataFrame({
        'log_diag_weight': log_diag_weights,
        'sq_photons': sq_phots,
        'coh_photons': coh_phots
    })
    df.to_csv(save_filename)

    # Plot sq photons
    color = next(color_cycler)['color']
    ax1.plot(log_diag_weights, sq_phots, color=color, label='M={}'.format(M))
    ax1.plot(log_diag_weights, [M * np.sinh(r_max) ** 2] * len(log_diag_weights),
             color=color, linestyle='--')

    # Plot coh photons
    ax2.plot(log_diag_weights, coh_phots, label='M={}'.format(M))

    # Plot sq/coh
    ax3.plot(log_diag_weights, sq_phots / coh_phots, label='M={}'.format(M))

ax1.legend()
fig1.savefig(dir+r'\sq_phot.pdf')
ax2.legend()
fig2.savefig(dir+r'\cl_phot.pdf')
ax3.legend()
fig3.savefig(dir+r'\sq_div_cl.pdf')

logging.info('Porsche')