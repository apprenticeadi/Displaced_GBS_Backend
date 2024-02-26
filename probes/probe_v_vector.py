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
vweights = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]  # edge activity
Ms = [10, 20, 30, 50, 100]

# <<<<<<<<<<<<<<<<<<< Fixed parameters (if any)  >>>>>>>>>>>>>>>>>>
x=-1
delta = 5
r_max = 0.75

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_v\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing the physical meaning of v vector. Mode M ={}. Fix max_degree={}, r_max={}, edge activity x={}. '
             'v=[1,...1]*vweight, where vweight from {}'.format(
    Ms, delta, r_max, x, vweights
))

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
# sq photons
fig1, ax1 = plt.subplots()
ax1.set_title('Average squeezed photons')
ax1.set_xlabel('v')
ax1.set_ylabel('n_sq')
ax1.set_xscale('log')
color_cycler = fig1.gca()._get_lines.prop_cycler

# cl photons
fig2, ax2 = plt.subplots()
ax2.set_title('Average classical photons')
ax2.set_xlabel('v')
ax2.set_ylabel('n_cl')
ax2.set_yscale('log')
ax2.set_xscale('log')

# sq/cl
fig3, ax3 = plt.subplots()
ax3.set_title('Photon number ratio squeezed/classical')
ax3.set_xlabel('v')
ax3.set_ylabel('n_sq/n_cl')
ax3.set_yscale('log')
ax3.set_xscale('log')

for M in Ms:

    # <<<<<<<<<<<<<<<<<<< Fixed parameters  >>>>>>>>>>>>>>>>>>
    half_gamma = np.ones(M)

    adj = RandomUtils.random_adj(M, delta)
    graph = MatchingGraph(adj, half_gamma=half_gamma, r_max=r_max, x=x)

    logging.info('M={}, half_gamma={}'.format(
        M, half_gamma,
    ))

    np.save(dir + r'\adj_M={}_delta={}'.format(M, delta), adj)

    save_filename = DFUtils.create_filename(
        dir + r'\M={}_delta={}.csv'.format(M, delta))
    sq_phots = []
    coh_phots = []

    for idx, vweight in enumerate(vweights):
        v = np.ones(M) * vweight

        graph.set_v(v)
        sq, displacement, U = graph.generate_experiment()

        logging.info('vweight={}, sq = {}, displacement = {}'.format(vweight, sq, displacement))
        np.save(dir+r'\U_M={}_delta={}'.format(M, delta), U)

        sq_phots.append(np.sum(np.sinh(sq) ** 2))
        coh_phots.append(np.sum(displacement * displacement.conjugate()).real)

    sq_phots = np.array(sq_phots)
    coh_phots = np.array(coh_phots)

    df = pd.DataFrame({
        'v': vweights,
        'sq_photons': sq_phots,
        'coh_photons': coh_phots
    })
    df.to_csv(save_filename)

    # Plot sq photons
    color = next(color_cycler)['color']
    ax1.plot(vweights, sq_phots, color=color, label='M={}'.format(M))
    ax1.plot(vweights, [M * np.sinh(r_max) ** 2] * len(vweights),
             color=color, linestyle='--')

    # Plot coh photons
    ax2.plot(vweights, coh_phots, label='M={}'.format(M))

    # Plot sq/coh
    ax3.plot(vweights, sq_phots / coh_phots, label='M={}'.format(M))

ax1.legend()
fig1.savefig(dir+r'\sq_phot.pdf')
ax2.legend()
fig2.savefig(dir+r'\cl_phot.pdf')
ax3.legend()
fig3.savefig(dir+r'\sq_div_cl.pdf')

logging.info('Aston Martin')