import numpy as np
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from strawberryfields.decompositions import takagi

from src.utils import MatrixUtils, DFUtils, LogUtils, RandomUtils
from src.gbs_matrix import GBSMatrix, GaussianMatrix
from src.adjacency_graph import MatchingGraph

# <<<<<<<<<<<<<<<<<<< Parameters  >>>>>>>>>>>>>>>>>>
gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]  # edge activity
Ms = [10, 20, 30, 50, 100]

# <<<<<<<<<<<<<<<<<<< Fixed parameters (if any)  >>>>>>>>>>>>>>>>>>
x=-1
delta = 5
r_max = 0.75

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_x\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing the physical meaning of half_gamma. Mode M ={}. Fix max_degree={}, r_max={}, edge activity x={}. '
             'half_gamma=[1,...1]*gamma, where gamma from {}'.format(
    Ms, delta, r_max, x, gammas
))

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
# sq photons
fig1, ax1 = plt.subplots()
ax1.set_title('Average squeezed photons')
ax1.set_xlabel('gamma')
ax1.set_ylabel('n_sq')
ax1.set_xscale('log')
color_cycler = fig1.gca()._get_lines.prop_cycler

# cl photons
fig2, ax2 = plt.subplots()
ax2.set_title('Average classical photons')
ax2.set_xlabel('gamma')
ax2.set_ylabel('n_cl')
ax2.set_yscale('log')
ax2.set_xscale('log')

# sq/cl
fig3, ax3 = plt.subplots()
ax3.set_title('Photon number ratio squeezed/classical')
ax3.set_xlabel('gamma')
ax3.set_ylabel('n_sq/n_cl')
ax3.set_yscale('log')
ax3.set_xscale('log')

for M in Ms:

    # <<<<<<<<<<<<<<<<<<< Fixed parameters  >>>>>>>>>>>>>>>>>>
    v = np.ones(M)

    adj = RandomUtils.random_adj(M, delta)
    graph = MatchingGraph(adj, v=v, r_max=r_max, x=x)

    logging.info('M={}, v={}'.format(
        M, v,
    ))

    np.save(dir + r'\adj_M={}_delta={}'.format(M, delta), adj)

    save_filename = DFUtils.create_filename(
        dir + r'\M={}_delta={}.csv'.format(M, delta))
    sq_phots = []
    coh_phots = []

    for idx, gamma in enumerate(gammas):
        half_gamma = np.ones(M) * gamma

        graph.set_half_gamma(half_gamma)
        sq, displacement, U = graph.generate_experiment()

        logging.info('gamma={}, sq = {}, displacement = {}'.format(gamma, sq, displacement))
        np.save(dir+r'\U_M={}_delta={}'.format(M, delta), U)

        sq_phots.append(np.sum(np.sinh(sq) ** 2))
        coh_phots.append(np.sum(displacement * displacement.conjugate()).real)

    sq_phots = np.array(sq_phots)
    coh_phots = np.array(coh_phots)

    df = pd.DataFrame({
        'gamma': gammas,
        'sq_photons': sq_phots,
        'coh_photons': coh_phots
    })
    df.to_csv(save_filename)

    # Plot sq photons
    color = next(color_cycler)['color']
    ax1.plot(gammas, sq_phots, color=color, label='M={}'.format(M))
    ax1.plot(gammas, [M * np.sinh(r_max) ** 2] * len(gammas),
             color=color, linestyle='--')

    # Plot coh photons
    ax2.plot(gammas, coh_phots, label='M={}'.format(M))

    # Plot sq/coh
    ax3.plot(gammas, sq_phots / coh_phots, label='M={}'.format(M))

ax1.legend()
fig1.savefig(dir+r'\sq_phot.pdf')
ax2.legend()
fig2.savefig(dir+r'\cl_phot.pdf')
ax3.legend()
fig3.savefig(dir+r'\sq_div_cl.pdf')

logging.info('Ferrari')