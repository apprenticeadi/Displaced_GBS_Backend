import numpy as np
import scipy
from scipy.optimize import minimize
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from strawberryfields.decompositions import takagi

from src.utils import RandomUtils, LogUtils
from src.interferometer import Interferometer, square_decomposition

# A design protocol that tries to target arbitrary sq/dis ratio.
# Keep everything real in this script, so DBD is real symmetric/ Hermitian

# <<<<<<<<<<<<<<<<<<< Size  >>>>>>>>>>>>>>>>>>
log_switch = True
M = 20
delta = 5
modes = list(range(M))

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
dir = r'..\Results\probe_sq_disp\{}'.format(time_stamp)
LogUtils.log_config(time_stamp='', dir=dir, filehead='log', module_name='', level=logging.INFO)

logging.info('Probing squeezing/displacement ratio in design protocol. '
             'Mode M ={}. Fix max_degree={}'.format(
    M, delta
))

results_df = pd.DataFrame(columns=['sq_target', 'sq_final', 'alpha_final', 'tanhr_target', 'v_init', 'half_gamma_init', 'v_final', 'half_gamma_final'])

# <<<<<<<<<<<<<<<<<<< Target  >>>>>>>>>>>>>>>>>>
sq_dis_ratio = 0.0001
logging.info('Target sq/dis ratio = {}'.format(sq_dis_ratio))

# sq_target = np.random.uniform(low=0.5, high=1.5, size=(M,))
sq_phot_target = np.sqrt(M) * sq_dis_ratio / (1 + sq_dis_ratio)  # We want total photon number to be on order of sqrt(M)
sq_target = np.arcsinh(np.sqrt(sq_phot_target / M)) * np.ones(M)
sq_target.sort()
tanhr_target = np.tanh(sq_target)

results_df['sq_target'] = sq_target
results_df['tanhr_target'] = tanhr_target
logging.info('Target sq photons = {}, \ntarget tanhr (flat) = {}'.format(
    sq_phot_target, tanhr_target[0]))

dis_phot_target = np.sqrt(M) / (1 + sq_dis_ratio)  # np.random.uniform(low=30, high=120)
alpha_guess = np.sqrt(dis_phot_target / M) * np.ones(M)

logging.info('Target disp photons={}'.format(dis_phot_target))

# <<<<<<<<<<<<<<<<<<< Graph  >>>>>>>>>>>>>>>>>>
x = -1 / 8
adj = RandomUtils.random_adj(M, delta)

G = nx.from_numpy_matrix(adj)  # create networkx graph
np.save(dir + r'\adj', adj)
logging.info('Edge activity = {}, adjacency matrix is saved separately.'.format(x))

def get_B(half_gamma, v):
    B = x * np.multiply(adj * half_gamma, half_gamma[:, np.newaxis]) + np.diag(v)
    return B

def get_tanhr(half_gamma, v):
    B = get_B(half_gamma, v)
    # tanhr, _ = takagi(DBD)

    tanhr = np.absolute(scipy.linalg.eigvalsh(B))
    tanhr.sort()

    return tanhr

def get_alpha(half_gamma, v):
    B = get_B(half_gamma, v)
    Id = np.eye(M)

    inv_covQ = np.block([[Id, -B], [-B, Id]])
    covQ = scipy.linalg.inv(inv_covQ)

    Gamma = np.concatenate([half_gamma, half_gamma])  # Again keeping everything real for now

    d_fock = covQ @ Gamma
    alpha = d_fock[:M]
    alpha.sort()

    return alpha

# <<<<<<<<<<<<<<<<<<< Construct the initial values  >>>>>>>>>>>>>>>>>>
half_gamma_init = alpha_guess
v_init = tanhr_target  # in small squeezing, v should be almost the same as tanhr
B_init = get_B(half_gamma_init, v_init)

# Rescale the initial values.
eigs_B_init = scipy.linalg.eigvalsh(B_init)
d_max = np.sqrt(max(tanhr_target) / np.absolute(max(eigs_B_init, key=abs)))  # keep initial D the same
d_vary = np.sqrt(tanhr_target / np.absolute(eigs_B_init))  # let initial D be different
half_gamma_init = d_max * half_gamma_init
v_init = d_max ** 2 * v_init
B_init = get_B(half_gamma_init, v_init)

# np.save(dir+r'\v_init', v_init)
# np.save(dir+r'\half_gamma_init', half_gamma_init)
np.save(dir+r'\B_init', B_init)
results_df['v_init'] = v_init
results_df['half_gamma_init'] = half_gamma_init


tanhr_init, U_init = takagi(B_init)
tanhr_init.sort()
alpha_init = get_alpha(half_gamma_init, v_init)
dis_phot_init = sum(np.absolute(alpha_init)**2)
logging.info('tanhr_init={}, \ndis_phot_init={}'.format(tanhr_init, dis_phot_init))


# <<<<<<<<<<<<<<<<<<< Construct the cost function  >>>>>>>>>>>>>>>>>>
def coeff(tanhr, dis_phot):
    coeff = 100 * dis_phot / np.average(tanhr)
    return coeff

def cost(gamma_v):
    half_gamma = gamma_v[:M]
    v = gamma_v[M:2*M]

    tanhr = get_tanhr(half_gamma, v)

    alpha = get_alpha(half_gamma, v)
    dis_phot = sum(np.absolute(alpha) ** 2)

    c = coeff(tanhr, dis_phot)

    if any(tanhr >= 1):
        return c *  max(tanhr) ** 2 + dis_phot_target ** 2 # Penalise tanhr larger than 1
    elif any(np.absolute(alpha) >= 1):
        return c * max(tanhr) ** 2 + dis_phot_target ** 2   # Penalise collisions caused by high collisions
    else:
        return c * sum(np.absolute(tanhr_target - tanhr) ** 2) / M + (dis_phot - dis_phot_target) ** 2 / M

# <<<<<<<<<<<<<<<<<<< Optimisation result  >>>>>>>>>>>>>>>>>>
result = minimize(cost, x0=np.concatenate([half_gamma_init, v_init]))

logging.info('Optimisation success = {}'.format(result.success))

gamma_v_final = result.x
half_gamma_final = gamma_v_final[:M]
v_final = gamma_v_final[M:2 * M]

results_df['v_final'] = v_final
results_df['half_gamma_final'] = half_gamma_final


B_final = get_B(half_gamma_final, v_final)
np.save(dir+r'\B_final', B_final)

tanhr, U = takagi(B_final)
tanhr.sort()
sq = np.arctanh(tanhr)
alpha = get_alpha(half_gamma_final, v_final)

np.save(dir+r'\U_final', U)
results_df['sq_final'] = sq
results_df['alpha_final'] = alpha

sq_phot_vector = np.sinh(sq) ** 2
sq_phot = sum(sq_phot_vector)

dis_phot_vector = np.absolute(alpha) ** 2
dis_phot = sum(dis_phot_vector)

logging.info('cost_sq={}'.format( coeff(tanhr, dis_phot) * sum(np.absolute(tanhr_target - tanhr) ** 2) / M))
logging.info('cost_dis={}'.format((dis_phot - dis_phot_target) ** 2 / M ))

logging.info('sq_phot={}'.format(sq_phot))
logging.info('dis_phot={}'.format(dis_phot))
logging.info('Final sq/dis ratio = {}'.format(sq_phot / dis_phot))

mean_photon_vector = dis_phot_vector + np.sum(np.square(np.absolute(U)) * sq_phot_vector, axis=1)

# <<<<<<<<<<<<<<<<<<< Plotting >>>>>>>>>>>>>>>>>>
fig = plt.figure("Constructed experiment", figsize=(8,8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(6, 6)

if M<=31:
    ax0 = fig.add_subplot(axgrid[0:4, :4])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title(r'Graph for $M=${}, $\Delta=${}'.format(M, delta))
    ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[4:, 4:])
ax1.bar(modes, mean_photon_vector)
ax1.set_xlabel('Mode')
ax1.set_xticks(modes)
ax1.set_ylabel(r'$\langle n_i \rangle$')
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-10, top=1e0)
ax1.set_title('Output photons')

ax2 = fig.add_subplot(axgrid[4:, :2])
ax2.bar(modes, sq_phot_vector)
ax2.set_title("Input squeezing")
ax2.set_xlabel("Mode")
ax2.set_ylabel(r'$\sinh(r_i)^2$')
ax2.set_xticks(modes)
ax2.set_yscale('log')
ax2.set_ylim(bottom=1e-10, top=1e0)

ax3 = fig.add_subplot(axgrid[4:, 2:4])
ax3.bar(modes, dis_phot_vector)
ax3.set_title('Input displacement')
ax3.set_xlabel('Mode')
ax3.set_ylabel(r'$|\alpha_i|^2$')
ax3.set_xticks(modes)
ax3.set_yscale('log')
ax3.set_ylim(bottom=1e-10, top=1e0)

ax4 = fig.add_subplot(axgrid[:4, 4:])
ax4.set_axis_off()
ax4.text(0, 0.6, 'Optimisation \nSuccess:{}'.format(result.success))
ax4.text(0, 0.5, r'$\sum_i \sinh(r_i)^2$={}'.format(np.format_float_positional(sq_phot, precision=3, unique=False, fractional=False, trim='k')))
ax4.text(0, 0.4, r'$\sum_i |\alpha_i|^2$={}'.format(np.format_float_positional(dis_phot, precision=3, unique=False, fractional=False, trim='k')))
ax4.text(0, 0.3, r'$\frac{\sum \sinh(r_i)^2}{\sum |\alpha_i|^2}$' + '={}'.format(np.format_float_positional(sq_dis_ratio, precision=3, unique=False, fractional=False, trim='k')))

fig.tight_layout()

plt.savefig(dir+r'\Experiment_{}.pdf'.format(result.success))

I = square_decomposition(U)
fig1 = I.draw()
plt.savefig(dir+r'\Interferometer.pdf')