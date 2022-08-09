import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import logging

from src.gbs_experiment import PureGBS
from src.utils import RandomUtils, MatrixUtils, LogUtils

M = 30
depth = 5

r = 0.2
rs = np.ones(M) * r

alpha = 10
alphas = np.ones(M) * alpha

gbs = PureGBS(M)

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_depth\M={}_depth={}_{}'.format(M, depth, time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing the connection between interferometer depth and graph structure. Mode M ={}. '
             'Fix interferometer depth ={}, '
             'squeezing params rs={}, displacements alphas={}. '.format(
    M, depth, rs, alphas,
))

# <<<<<<<<<<<<<<<<<<< Interferometer  >>>>>>>>>>>>>>>>>>
logging.info('Generating interferometer time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))
I = RandomUtils.random_interferometer(M, depth)
logging.info('Calculating U matrix time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))
U = I.calculate_transformation()
logging.info('Finish calculating U matrix time={}'.format(datetime.datetime.now().strftime("%H-%M-%S.%f")))

np.save(dir + r'\U.npy', U)

if M <=31:
    fig0 = I.draw()
    fig0.savefig(dir+r'\interferometer.pdf')

# <<<<<<<<<<<<<<<<<<< Experiment  >>>>>>>>>>>>>>>>>>
gbs.add_interferometer(U)
gbs.add_squeezing(rs)
gbs.add_coherent(alphas)

# <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>
x = gbs.generate_weighted_adj()
np.save(dir+r'\x.npy', x)

adj = (x != 0)
# adj.astype(int)

# <<<<<<<<<<<<<<<<<<< graph  >>>>>>>>>>>>>>>>>>

G = nx.from_numpy_matrix(adj)

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(10, 10)

if M<=31:
    ax0 = fig.add_subplot(axgrid[0:5, :5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Graph for M={}, depth={}".format(M, depth))
    ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[5:, :5])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[5:, 5:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

# fig.tight_layout()
# fig.savefig(dir + r'\graph_M={}_d={}.pdf'.format(M, depth))

# <<<<<<<<<<<<<<<<<<< Edge activities  >>>>>>>>>>>>>>>>>>

# dmax = min(M, 4*depth-3)

x = x.flatten()
x_max = abs(max(x, key=abs))
x_min = abs(min(x[np.nonzero(x)], key=abs))
logxmin = int(np.log10(x_min) - 0.5)
disc_rad = 1 / (np.e**2 * dmax)
lograd = int(np.log10(disc_rad) - 0.5)
axislim = max(x_max, disc_rad) +1

# fig3 = plt.figure('Edge activities', figsize=(6, 6))

# ax3 = fig3.add_subplot()
ax3 = fig.add_subplot(axgrid[0:4, 6:])
ax3.plot(x.real, x.imag, marker='.', linestyle='None')
ax3.grid(True, which='both')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
circ = plt.Circle((0, 0), radius=disc_rad, edgecolor='b', facecolor='None', label=r'$1/(\Delta e^2)$')
ax3.add_patch(circ)
ax3.set_title('Edge activities for r={}, alpha={}'.format(r, alpha))
ax3.set_xlabel('Re(x)')
ax3.set_ylabel('Imag(x)')
ax3.set_xlim(-axislim, axislim)
ax3.set_ylim(-axislim, axislim)
ax3.set_xscale('symlog', linthresh=x_min)
ax3.set_yscale('symlog', linthresh=x_min)
ax3.set_xticks([-1, -10**lograd, -logxmin, logxmin, 10**lograd, 1])
ax3.set_yticks([-1, -10**lograd, -logxmin, logxmin, 10**lograd, 1])
ax3.legend()

# fig3.savefig(dir+r'\edge_activity_r={}_a={}.pdf'.format(r, alpha))

fig.tight_layout()
fig.savefig(dir + r'\graph_M={}_d={}.pdf'.format(M, depth))