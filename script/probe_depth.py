import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from src.gbs_experiment import PureGBS
from src.utils import MatrixUtils

M = 30
depth = 10

r = random.random()
rs = np.ones(M) * r

alpha = 1
alphas = np.ones(M) * alpha

gbs = PureGBS(M)

I = PureGBS.random_interferometer(M, depth)
U = I.calculate_transformation()

I.draw()

gbs.add_interferometer(U)
gbs.add_squeezing(rs)
gbs.add_coherent(alphas)

# <<<<<<<<<<<<<<<<<<< B matrix  >>>>>>>>>>>>>>>>>>
B = gbs.calc_B()
B = MatrixUtils.filldiag(B, np.zeros(M))
gamma = gbs.calc_Gamma()[:M]

# <<<<<<<<<<<<<<<<<<< edge activities  >>>>>>>>>>>>>>>>>>
x = B / np.outer(gamma, gamma)

# <<<<<<<<<<<<<<<<<<< adjacency matrix  >>>>>>>>>>>>>>>>>>
adj = (x != 0)
adj.astype(int)

# <<<<<<<<<<<<<<<<<<< graph  >>>>>>>>>>>>>>>>>>

G = nx.from_numpy_matrix(adj)

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(5, 4)

ax0 = fig.add_subplot(axgrid[0:3, :])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Graph for M={}, depth={}".format(M, depth))
ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[3:, :2])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[3:, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()


