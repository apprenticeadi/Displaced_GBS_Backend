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
xs = [-1000, -500, -100, -50, -10, -5, -4, -3, -2, -1, -0.5, -0.25, -0.125, -0.1, -0.09, -0.08, -0.07, -0.06, -0.05,
      -0.04, -0.03, -0.02, -0.01, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]  # edge activity
Ms = [10, 20, 30, 50, 100]

# <<<<<<<<<<<<<<<<<<< Fixed parameters (if any)  >>>>>>>>>>>>>>>>>>
delta = 5
bound = - 1 / (4*delta - 4)
r_max = 0.75

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_x\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing the physical meaning of x. Mode M ={}, edge activity xs={}. Fix max_degree={}, r_max={}'.format(
    Ms, xs, delta, r_max
))

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
# sq photons
fig1, ax1 = plt.subplots()
ax1.set_title('Average squeezed photons')
ax1.set_xlabel('x')
ax1.set_ylabel('n_sq')
ax1.set_xscale('symlog')
color_cycler = fig1.gca()._get_lines.prop_cycler

# cl photons
fig2, ax2 = plt.subplots()
ax2.set_title('Average classical photons')
ax2.set_xlabel('x')
ax2.set_ylabel('n_cl')
ax2.set_yscale('log')
ax2.set_xscale('symlog')

# sq/cl
fig3, ax3 = plt.subplots()
ax3.set_title('Photon number ratio squeezed/classical')
ax3.set_xlabel('x')
ax3.set_ylabel('n_sq/n_cl')
ax3.set_yscale('log')
ax3.set_xscale('symlog')
ax3.axvline(x=bound, ls=':', label='-1/(4*delta-4)')

for M in Ms:

    # <<<<<<<<<<<<<<<<<<< Fixed parameters  >>>>>>>>>>>>>>>>>>
    half_gamma = np.ones(M)
    v = np.ones(M)

    bound = -1 / (4 * (delta - 1))

    adj = RandomUtils.random_adj(M, delta)
    graph = MatchingGraph(adj, half_gamma=half_gamma, v=v, r_max=r_max)

    logging.info('For M={}, construct adj=\n'
                 '{}\n'
                 'Set half_gamma={}, v={}'.format(
        M, adj, half_gamma, v,
    ))

    np.save(dir + r'\adj_M={}_delta={}'.format(M, delta), adj)

    save_filename = DFUtils.create_filename(
        dir + r'\M={}_delta={}.csv'.format(M, delta))
    sq_phots = []
    coh_phots = []

    for idx, x in enumerate(xs):
        logging.info('x={}'.format(x))
        graph.set_x(x)
        sq, displacement, U = graph.generate_experiment()

        logging.info('sq = {}, displacement = {}, U={}'.format(sq, displacement, U))

        sq_phots.append(np.sum(np.sinh(sq) ** 2))
        coh_phots.append(np.sum(displacement * displacement.conjugate()).real)

    sq_phots = np.array(sq_phots)
    coh_phots = np.array(coh_phots)

    df = pd.DataFrame({
        'x': xs,
        'sq_photons': sq_phots,
        'coh_photons': coh_phots
    })
    df.to_csv(save_filename)

    # Plot sq photons
    color = next(color_cycler)['color']
    ax1.plot(xs, sq_phots, color=color, label='M={}'.format(M))
    ax1.plot(xs, [M * np.sinh(r_max) ** 2] * len(xs),
             color=color, linestyle='--')

    # Plot coh photons
    ax2.plot(xs, coh_phots, label='M={}'.format(M))

    # Plot sq/coh
    ax3.plot(xs, sq_phots / coh_phots, label='M={}'.format(M))

ax1.legend()
fig1.savefig(dir+r'\sq_phot.pdf')
ax2.legend()
fig2.savefig(dir+r'\cl_phot.pdf')
ax3.legend()
fig3.savefig(dir+r'\sq_div_cl.pdf')

logging.info('Ferrari')