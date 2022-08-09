from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
import numpy as np

from src.gbs_experiment import PureGBS, TakagiGBS

def inline_circuit(M, B, half_gamma, v=None):
    circuit = PureGBS(M)

    B = np.asarray(B)
    half_gamma = np.asarray(half_gamma)
    if v is None:
        v = B.diagonal()

    circuit.add_displacement(half_gamma)
    circuit.add_squeezing(v)

    for j in range(M):
        for k in range(j + 1, M):
            Bjk = B[j, k]

            circuit.add_two_mode_squeezing(Bjk, (j,k))

    return circuit

M = 4
U = unitary_group.rvs(M)
r = np.asarray([0.5] * M)
alpha = np.asarray([2] * M)

gbs = TakagiGBS(M)
gbs.add_squeezing(r)
gbs.add_interferometer(U)
gbs.add_displacement(alpha)
B = gbs.calc_B()
half_gamma = gbs.calc_half_gamma()

mu1, cov1 = gbs.state_fock()

inline = inline_circuit(M, B, half_gamma)
mu2, cov2 = inline.state_fock()




