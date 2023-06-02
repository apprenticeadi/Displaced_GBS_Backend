import numpy as np
import strawberryfields as sf
from strawberryfields.tdm import get_mode_indices, borealis_gbs, full_compile
from strawberryfields.ops import Sgate, Rgate, BSgate, LossChannel, MeasureFock
import json
from thewalrus.quantum import Amat
import networkx as nx
import matplotlib.pyplot as plt

hbar = 1
sf.hbar = hbar

device_spec = json.load(open(r'..\borealis_device_specification.json'))
device_cert = json.load(open(r'..\borealis_device_certificate.json'))
device = sf.Device(spec=device_spec, cert=device_cert)

modes = 216

# r = [1.234] * modes
#
# min_phi, max_phi = 0, 2 * np.pi   # Arguments for rotation gates
# min_T, max_T = 0.4, 0.6  # Beamsplitter intensity transmission to obtain denser adjacency matrix
#
# # rotation-gate parameters
# phi_0 = np.random.uniform(low=min_phi, high=max_phi, size=modes)
# phi_1 = np.random.uniform(low=min_phi, high=max_phi, size=modes)
# phi_2 = np.random.uniform(low=min_phi, high=max_phi, size=modes)
#
# # beamsplitter parameters
# T_0 = np.random.uniform(low=min_T, high=max_T, size=modes)
# T_1 = np.random.uniform(low=min_T, high=max_T, size=modes)
# T_2 = np.random.uniform(low=min_T, high=max_T, size=modes)
# alpha_0 = np.arccos(np.sqrt(T_0))
# alpha_1 = np.arccos(np.sqrt(T_1))
# alpha_2 = np.arccos(np.sqrt(T_2))

#
# # the travel time per delay line in time bins
# delay_0, delay_1, delay_2 = 1, 6, 36
#
# # set the first beamsplitter arguments to 'T=1' ('alpha=0') to fill the
# # loops with pulses
# alpha_0[:delay_0] = 0.0
# alpha_1[:delay_1] = 0.0
# alpha_2[:delay_2] = 0.0
#
# gate_args = {
#     "Sgate": r,
#     "loops": {
#         0: {"Rgate": phi_0.tolist(), "BSgate": alpha_0.tolist()},
#         1: {"Rgate": phi_1.tolist(), "BSgate": alpha_1.tolist()},
#         2: {"Rgate": phi_2.tolist(), "BSgate": alpha_2.tolist()},
#     },
# }
#
# gate_args_list = full_compile(gate_args, device)
#

gate_args_list = borealis_gbs(device, modes=modes, squeezing="high")

delays = [1, 6, 36]
vac_modes = sum(delays)

n, N = get_mode_indices(delays)

prog = sf.TDMProgram(N)
with prog.context(*gate_args_list) as (p, q):
    # Sgate(p[0]) | q[n[0]]
    for i in range(len(delays)):
        Rgate(p[2 * i + 1]) | q[n[i]]
        BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
    # MeasureFock() | q[0]

prog.space_unroll(1)
prog = prog.compile(compiler='passive')

T_tot = prog.circuit[0].op.p[0]
T = T_tot[vac_modes:, vac_modes:]



# compile_options = {
#     "device": device,
#     "realistic_loss": False,
# }
#
# run_options = {
#     "shots": None,
#     "crop": True,
#     "space_unroll": True,
# }
#
# eng_sim = sf.Engine(backend="gaussian")
# # results_sim = eng_sim.run(prog, **run_options, compile_options=compile_options)
# results_sim = eng_sim.run(prog, **run_options)
# state = results_sim.state
# cov = state.cov()
# mu = state.means()
#
# A = Amat(cov, hbar= hbar)
# B = A[:modes, :modes]
#
# adj = (B != 0)
# G = nx.from_numpy_matrix(adj)
# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# dmax = max(degree_sequence)
#
# fig = plt.figure("Degree of Borealis", figsize=(8, 8))
# # Create a gridspec for adding subplots of different sizes
# axgrid = fig.add_gridspec(5, 10)
#
# ax1 = fig.add_subplot(axgrid[:, :5])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")
#
# ax2 = fig.add_subplot(axgrid[:, 5:])
# ax2.bar(*np.unique(degree_sequence, return_counts=True))
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("# of Nodes")
