import numpy as np
import scipy.stats
from scipy.stats import unitary_group
from numpy.polynomial import Polynomial
import networkx as nx
import time
import numba
import logging
import matplotlib
import matplotlib.pyplot as plt
# from thewalrus import hafnian

from src.utils import MatrixUtils, LogUtils, DFUtils
import datetime

plt.show()
plt.ion()
matplotlib.use('TkAgg')

def generate_tildeX(N, K, repeats, mean=0, stddev=1/np.sqrt(2)):

    tildeXs = np.zeros((repeats, N, N), dtype=np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N, K)) + 1j * np.random.normal(mean, stddev, (N, K))
        tildeX = (X @ X.T) / np.outer(np.sum(X, axis=1), np.sum(X, axis=1))
        assert tildeX.shape == (N,N)
        tildeXs[i,:,:] = tildeX

    return tildeXs

def generate_X(N, K, repeats, mean=0, stddev=1/np.sqrt(2)):
    assert N == K  # the GBS matrix has to be a square matrix for a fully connected graph.
    Xs = np.zeros((repeats, N, K), dtype = np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N,K)) + 1j * np.random.normal(mean, stddev, (N,K))
        Xs[i, :, :] = X

    return Xs

def generate_symX(N, K, repeats, mean=0, stddev=1/np.sqrt(2)):

    symXs = np.zeros((repeats, N,N), dtype=np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N,K)) + 1j * np.random.normal(mean, stddev, (N,K))
        symX = X @ X.T

        symXs[i, :, :] = symX

    return symXs

def generate_tildesubU(N, K, M, repeats):

    assert N <= M
    assert K <= M
    subUs = np.zeros((repeats, N, N), dtype=np.complex128)
    for i in range(repeats):
        # generate a random M x M unitary matrix U
        U = unitary_group.rvs(M)

        # get a random N x K submatrix of U
        rows = np.random.choice(M, N, replace=False)
        cols = np.random.choice(M, K, replace=False)
        subU = U[rows, :][:, cols]
        tildesubU = (subU @ subU.T) / np.outer(np.sum(subU, axis=1), np.sum(subU, axis=1))

        subUs[i, :, :] = tildesubU

    return subUs

@numba.njit(parallel=True)
def find_roots(coeffs):
    num_pol, num_roots = coeffs.shape
    num_roots -= 1

    all_roots = np.zeros(num_pol * num_roots, dtype=np.complex128)
    for id in numba.prange(num_pol):
        roots = np.roots(coeffs[id][::-1])
        all_roots[id*num_roots: (id+1)*num_roots] = roots

    return all_roots


# <<<<<<<<<<<<<<<<<<< Basic Parameters  >>>>>>>>>>>>>>>>>>
Ns = [12] # [4, 8, 12, 14]  # np.arange(4, 16)
Ks = 144 * np.ones(len(Ns), dtype=int)  # Ns
M = 144
mean = 0
stddev = 1 / np.sqrt(2)
repeats = 10000
roots_dict = {}
distrib = 'sub_unitary' # 'sym_gaussian' # 'gaussian'  # 'tilde_gaussian'  # 'sub_unitary

save_fig = True
min_roots = np.zeros(len(Ns), dtype=float)
max_roots = np.zeros(len(Ns), dtype=float)
median_roots = np.zeros(len(Ns), dtype=float)

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = fr'..\Results\roots_matching_polynomial\{repeats}rep_{time_stamp}'
LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)

if distrib == 'sub_unitary':
    logging.info(f'In this script, I calculate the zeros of f(z)=lhaf(A z, 1), where A is {distrib} distribution '
                 f'defined using rescaled submatrices of a random unitary matrix of size {M}. The submatrices have '
                 f'dimensions N*K, and rescaled by '
                 f'(subU @ subU.T) / np.outer(np.sum(subU, axis=1), np.sum(subU, axis=1))'
                 f'N taken from Ns={Ns}, and corresponding K is {Ks}. ')
else:
    logging.info(f'In this script, I calculate the zeros of f(z)=lhaf(A z, 1), where A is {distrib} distribution '
                 f'defined using i.i.d Gaussians with mean={mean} and stddev of'
                 f'real (imaginary) part = {stddev}. N taken from Ns={Ns}, and corresponding K is {Ks}. ')

logging.info(f'For each N, matrix A is generated for repeats={repeats} times, and int(N/2) roots are calculated for each '
             f'matrix A. ')
logging.info(f'The matrices A are stored as [repeats, N, N] arrays, the matching polynomial coefficients are stored as'
             f'[repeats, int(N/2)+1] arrays, where the 0-th column is the zero-th order coefficient etc. '
             f'The roots are stored as [repeats, int(N/2)] arrays. ')

# <<<<<<<<<<<<<<<<<<< Calculations  >>>>>>>>>>>>>>>>>>
for i_N, N in enumerate(Ns):

    K = Ks[i_N]  # number of non-vacuum modes. K defines the distribution.

    k_max = int(N / 2)  # maximum degree of matching polynomial, also number of roots

    if distrib == 'tilde_gaussian':
        As = generate_tildeX(N, K, repeats, mean=mean, stddev=stddev)
    elif distrib == 'gaussian':
        As = generate_X(N, K, repeats, mean=mean, stddev=stddev)
    elif distrib == 'sym_gaussian':
        As = generate_symX(N, K, repeats, mean=mean, stddev=stddev)
    elif distrib == 'sub_unitary':
        As = generate_tildesubU(N, K, M, repeats)
    else:
        raise ValueError(f'distribution {distrib} not recognized')


    np.save(DFUtils.create_filename(results_dir + fr'\raw\N={N}_K={K}_tildeXs.npy'), As)

    coeffs = np.zeros((repeats, k_max + 1), dtype=np.complex128)
    coeffs[:, 0] = 1

    t1 = time.time()
    # Construct graph
    G = nx.complete_graph(N)  # this shouldn't have self loops
    L = nx.line_graph(G)
    LC = nx.complement(L)
    # enumerate cliques in LC, does not include empty set.
    for clique in nx.enumerate_all_cliques(LC):
        # each clique is a list of nodes in LC. they have the form [(u1,v1), (u2,v2), ..., (uk, vk)] for a k-clique.
        k = len(clique)
        cs = np.ones(repeats, dtype=np.complex128)
        for node in clique:
            cs = cs * As[:, node[0], node[1]]

        coeffs[:, k] += cs
    t2 = time.time()

    logging.info(f'N={N}, time={t2 - t1} to construct {repeats} matching polynomials')

    np.save(DFUtils.create_filename(results_dir + fr'\raw\N={N}_K={K}_mp_coeffs.npy'), coeffs)

    # # Test function
    # lhaf_vals = np.zeros(repeats, dtype=np.complex128)
    # for i in range(repeats):
    #     tildeX = tildeXs[i]
    #     lhaf_vals[i] = hafnian(tildeX + np.diag(np.ones(N) - np.diag(tildeX)), loop = True)
    #
    # print(np.allclose(np.sum(coeffs, axis=1), lhaf_vals))

    roots_N = find_roots(coeffs)
    t3 = time.time()

    roots_dict[N] = roots_N

    min_root = np.min(np.abs(roots_N))
    max_root = np.max(np.abs(roots_N))
    median_root = np.median(np.abs(roots_N))
    min_roots[i_N] = min_root
    max_roots[i_N] = max_root
    median_roots[i_N] = median_root

    np.save(DFUtils.create_filename(results_dir + fr'\N={N}_K={K}_roots.npy'), roots_N)

    logging.info(f'N={N}, time={t3 - t2} for to find roots for {repeats} matching polynomials')


np.save(results_dir + fr'\median_abs_roots.npy', median_roots)
np.save(results_dir + fr'\min_abs_roots.npy', min_roots)
np.save(results_dir + fr'\max_abs_roots.npy', max_roots)

# <<<<<<<<<<<<<<<<<<< Plotting  >>>>>>>>>>>>>>>>>>
log_linthresh = int(np.log(np.min(min_roots))) - 1
linthresh = 10 ** (log_linthresh)
axis_log_lim = int(np.log(np.max(max_roots))) + 1
axis_lim = 10 ** (axis_log_lim)
half_axis_ticks = 10 ** np.arange(log_linthresh, axis_log_lim+1, step=2, dtype=np.float64)
axis_ticks = np.concatenate([-half_axis_ticks, half_axis_ticks, [0]])

for i_N, N in enumerate(Ns):
    save_name = results_dir + fr'\plots\N={N}_K={K}.png'

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in roots_dict[N]]
    roots_imag = [np.imag(r) for r in roots_dict[N]]

    plt.figure(f'N={N},K={K}')

    # Remove plot boundaries (spines)
    ax = plt.gca()  # Get the current Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Create solid lines for the real and imaginary axes
    plt.axhline(0, color='black', linewidth=1)  # Real axis
    plt.axvline(0, color='black', linewidth=1)  # Imaginary axis

    # Scatter plot roots
    plt.scatter(roots_real, roots_imag, marker='x', s=10)

    # draw red circle with radius median_roots[i_N]
    circle = plt.Circle((0, 0), radius=median_roots[i_N], color='red', fill=False, label=rf'$|z|=${median_roots[i_N]:.3g}')
    ax.add_patch(circle)

    # Set axes
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)

    plt.xscale('symlog', linthresh=linthresh)
    plt.yscale('symlog', linthresh=linthresh)

    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    ax.legend()

    if save_fig:
        plt.savefig(DFUtils.create_filename(save_name))

    plt.show()


plt.figure('min and max |root|')
plt.plot(Ns, min_roots, '.', label='min')
plt.plot(Ns, max_roots, '.', label='max')
#TODO: bootstrap estimate this
plt.plot(Ns, median_roots, '.', label='median')
plt.xlabel(r'$N$')
plt.yscale('log')
plt.legend()
if save_fig:
    plt.savefig(DFUtils.create_filename(results_dir + r'\plots\min_and_max_root.png'))

plt.show()