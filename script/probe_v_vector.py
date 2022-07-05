import numpy as np
import logging
import datetime
import pandas as pd

from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils, DFUtils
from src.random_graph import RandomGraph
from src.gbs_matrices import GraphMatrices, GaussianMatrices
from src.log_utils import LogUtils

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
LogUtils.log_config('', level=logging.INFO)
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

for M in [5, 8, 10, 13, 15, 18, 20]:
    # <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>

    delta = max(3, int(0.5 * M))  # maximum degree
    graph_G = RandomGraph(M=M, max_degree=delta)

    logging.info('')
    logging.info('M={} and max degree={}'.format(M, delta))
    logging.info('Graph adjacency matrix is \n{}'.format(graph_G.adjacency_matrix()))

    # <<<<<<<<<<<<<<<<<<< Parameters  >>>>>>>>>>>>>>>>>>
    x = -1  # edge activity
    half_gamma = 1 - np.random.uniform(low=0, high=1, size=M)  # it cannot take 0 but can take 1
    r_max = 1  # maximum squeezing available in experiment

    logging.info('edge activity x = {}, half_gamma={}, max squeezing parameter={}'.format(x, half_gamma, r_max))

    log_diag_weights = list(range(1, 6))

    # <<<<<<<<<<<<<<<<<<< Data file  >>>>>>>>>>>>>>>>>>
    if x < 0:
        complexity = 'hard'
    else:
        complexity = 'easy'

    save_filename = DFUtils.create_filename(
        r'..\Results\probe_even_v\{}\M={}_delta={}_{}'.format(complexity, M, delta, time_stamp))

    df = pd.DataFrame(columns=['log_diag_weight', 'sq_photons', 'coh_photons'])

    for idx, log_diag_weight in enumerate(log_diag_weights):
        v = np.ones(M) * 10 ** log_diag_weight  # np.random.uniform(low=0, high=1, size=M) * 100000
        logging.info('For diag_weight={}, v = {}'.format(10 ** log_diag_weight, v))

        Bmat = graph_G.generate_Bmatrix(x, half_gamma)
        Bmat = MatrixUtils.filldiag(Bmat, v)

        eigs_B = np.linalg.eigvalsh(Bmat)
        c_factor = np.tanh(r_max) / abs(max(eigs_B, key=abs))

        cB = c_factor * Bmat
        gamma = np.sqrt(c_factor) * np.concatenate([half_gamma, half_gamma.conjugate()])

        Amat = GraphMatrices.pure_A_from_B(cB)
        mu_fock = GaussianMatrices.mu_fock_from_A(Amat, gamma)
        tanhr, U = takagi(cB)

        sq = np.arctanh(tanhr)
        displacement = mu_fock[:M]
        logging.info('sq = {}, displacement = {}'.format(sq, displacement))

        sq_photons = np.sum(np.sinh(sq) ** 2)
        coh_photons = np.sum(displacement * displacement.conjugate()).real

        df.loc[idx] = {
            'log_diag_weight': log_diag_weight,
            'sq_photons': sq_photons,
            'coh_photons': coh_photons
        }

    df.to_csv(save_filename)
