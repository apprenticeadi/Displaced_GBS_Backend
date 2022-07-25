import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.gbs_experiment import PureGBS
from src.utils import RandomUtils, MatrixUtils

M = 20
depth = 5

r = 0.75
rs = np.ones(M) * r

alpha = 1
alphas = np.ones(M) * alpha

gbs = PureGBS(M)

# <<<<<<<<<<<<<<<<<<< Interferometer  >>>>>>>>>>>>>>>>>>
I = RandomUtils.random_interferometer(M, depth)
U = I.calculate_transformation()
#
# I.draw()

# <<<<<<<<<<<<<<<<<<< Experiment  >>>>>>>>>>>>>>>>>>
gbs.add_interferometer(U)
gbs.add_squeezing(rs)
gbs.add_coherent(alphas)

# <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>
x = gbs.generate_weighted_adj()

adj = (x != 0)
adj.astype(int)

# <<<<<<<<<<<<<<<<<<< graph  >>>>>>>>>>>>>>>>>>

G = nx.from_numpy_matrix(adj)

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(6, 6)

ax0 = fig.add_subplot(axgrid[0:3, :3])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Graph for M={}, depth={}".format(M, depth))
ax0.set_axis_off()



x = x.flatten()
x_max = abs(max(x, key=abs))
axislim = max(1, x_max+1)

ax3 = fig.add_subplot(axgrid[0:3, 3:])
ax3.plot(x.real, x.imag, marker='.', linestyle='None')
ax3.grid(True, which='both')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
circ = plt.Circle((0, 0), radius=0.125, edgecolor='b', facecolor='None')
ax3.add_patch(circ)
ax3.set_title('Edge activities')
ax3.set_xlabel('Re(x)')
ax3.set_ylabel('Imag(x)')
ax3.set_xlim(-axislim, axislim)
ax3.set_ylim(-axislim, axislim)

ax1 = fig.add_subplot(axgrid[3:, :3])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[3:, 3:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()


