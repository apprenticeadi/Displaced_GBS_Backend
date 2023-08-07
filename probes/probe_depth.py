import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import logging

from src.gbs_experiment import sduGBS
from src.utils import RandomUtils, MatrixUtils, LogUtils

M = 31
depth = 5

r = 0.1
rs = np.ones(M) * r

beta = 10
betas = np.ones(M) * beta

gbs = sduGBS(M)

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_depth\M={}_depth={}_{}'.format(M, depth, time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing the connection between interferometer depth and graph structure. Mode M ={}. '
             'Fix interferometer depth ={}, '
             'squeezing params rs={}, displacements betas={}. '.format(
    M, depth, rs, betas,
))

# <<<<<<<<<<<<<<<<<<< Interferometer  >>>>>>>>>>>>>>>>>>
logging.info('Generating interferometer time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))
I = RandomUtils.random_interferometer(M, depth)
logging.info('Calculating U matrix time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))
U = I.calculate_transformation()
logging.info('Finish calculating U matrix time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))

np.save(dir + r'\U.npy', U)

if M <=31:
    fig0 = I.draw(show_params=False)
    fig0.savefig(dir+r'\interferometer.pdf')

# <<<<<<<<<<<<<<<<<<< Experiment  >>>>>>>>>>>>>>>>>>
gbs.add_squeezing(rs)
gbs.add_displacement(betas)
gbs.add_interferometer(U)

# <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>
tildeB = gbs.generate_weighted_adj()  # tilde B matrix
np.save(dir+r'\tildeB.npy', tildeB)

adj = (tildeB != 0)  # unweighted graph adjacency matrix
# adj.astype(int)

# <<<<<<<<<<<<<<<<<<< graph  >>>>>>>>>>>>>>>>>>
G = nx.from_numpy_matrix(adj)

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Graph")
# Create a gridspec for adding subplots of different sizes


if M<=31:
    ax0 = fig.add_subplot(111)
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    # ax0.set_title("Graph for M={}, depth={}".format(M, depth))
    ax0.set_axis_off()

# ax1 = fig.add_subplot(axgrid[5:, :5])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")

fig2 = plt.figure('Degree histogram')
ax2 = fig2.add_subplot(111)
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("#Vertices")

# fig.tight_layout()
# fig.savefig(dir + r'\graph_M={}_d={}.pdf'.format(M, depth))

# <<<<<<<<<<<<<<<<<<< Edge activities  >>>>>>>>>>>>>>>>>>

# dmax = min(M, 4*depth-3)

x = tildeB.flatten()
x_max = abs(max(x, key=abs))
x_min = abs(min(x[np.nonzero(x)], key=abs))
logxmin = int(np.log10(x_min) - 0.5)
disc_rad = 1 / (np.e * (2* dmax -1 ))
lograd = int(np.log10(disc_rad) - 0.5)
axislim = max(x_max, disc_rad) +1

# fig3 = plt.figure('Edge activities', figsize=(6, 6))

# ax3 = fig3.add_subplot()
fig3 = plt.figure('tildeB_ij')
ax3 = fig3.add_subplot(111)
ax3.plot(x.real, x.imag, marker='.', linestyle='None')
ax3.grid(True, which='both')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
circ = plt.Circle((0, 0), radius=disc_rad, edgecolor='b', facecolor='None', label=r'$1/e(2\Delta-1)$')
ax3.add_patch(circ)
ax3.set_title(r'$\tilde{B}_{ij}$' + fr' for r={r}, beta={beta}')
ax3.set_xlabel(r'$Re(\tilde{B}_{ij})$')
ax3.set_ylabel(r'$Im(\tilde{B}_{ij})$')
ax3.set_xlim(-axislim, axislim)
ax3.set_ylim(-axislim, axislim)
ax3.set_xscale('symlog', linthresh=x_min)
ax3.set_yscale('symlog', linthresh=x_min)
ax3.set_xticks([-1, -10**lograd, -logxmin, logxmin, 10**lograd, 1])
ax3.set_yticks([-1, -10**lograd, -logxmin, logxmin, 10**lograd, 1])
ax3.legend()

# fig3.savefig(dir+r'\edge_activity_r={}_a={}.pdf'.format(r, alpha))

fig.savefig(dir + rf'\graph_M={M}_depth={depth}.png')
fig2.savefig(dir + rf'\Degree_histogram_M={M}_depth={depth}.png')
fig3.savefig(dir + rf'\tildeB_ij_M={M}_depth={depth}.png')