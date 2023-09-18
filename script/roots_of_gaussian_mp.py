import numpy as np
from numpy.polynomial import Polynomial
import networkx as nx
import time
import numba
import logging
import matplotlib.pyplot as plt
from thewalrus import hafnian

from src.utils import MatrixUtils, LogUtils, DFUtils
import datetime


def generate_tildeX(N, repeats, mean=0, stddev=1/np.sqrt(2)):

    tildeXs = np.zeros((repeats, N, N), dtype=np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N, N)) + 1j * np.random.normal(mean, stddev, (N, N))
        tildeX = (X @ X.T) / np.outer(np.sum(X, axis=1), np.sum(X, axis=1))

        tildeXs[i,:,:] = tildeX

    return tildeXs

def generate_X(N, repeats, mean=0, stddev=1/np.sqrt(2)):

    Xs = np.zeros((repeats, N, N), dtype = np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N,N)) + 1j * np.random.normal(mean, stddev, (N,N))
        Xs[i, :, :] = X

    return Xs

def generate_symX(N, repeats, mean=0, stddev=1/np.sqrt(2)):

    symXs = np.zeros((repeats, N,N), dtype=np.complex128)
    for i in range(repeats):
        X = np.random.normal(mean, stddev, (N,N)) + 1j * np.random.normal(mean, stddev, (N,N))
        symX = X @ X.T

        symXs[i, :, :] = symX

    return symXs

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
Ns = [20] # np.arange(4, 16)
mean = 0
stddev = 1 # 1 / np.sqrt(2)
repeats = 1
roots_dict = {}
distrib = 'tilde_gaussian' # 'sym_gaussian' # 'gaussian'  # 'tilde_gaussian'

save_fig = True
min_roots = np.zeros(len(Ns), dtype=float)
max_roots = np.zeros(len(Ns), dtype=float)
median_roots = np.zeros(len(Ns), dtype=float)

# <<<<<<<<<<<<<<<<<<< Logging  >>>>>>>>>>>>>>>>>>
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = fr'..\Results\roots_matching_polynomial\{repeats}rep_{time_stamp}'
LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)

logging.info(f'In this script, I calculate the zeros of f(z)=lhaf(A z, 1), where A is {distrib} distribution '
             f'defined using i.i.d Gaussians with mean={mean} and stddev of'
             f'real (imaginary) part = {stddev}. N taken from Ns={Ns}. For each N, tildeX is generated for '
             f'repeats={repeats} times, and int(N/2) roots are calculated for each tildeX. ')
logging.info(f'The tildeXs are stored as [repeats, N, N] arrays, the matching polynomial coefficients are stored as'
             f'[repeats, int(N/2)+1] arrays, where the 0-th column is the zero-th order coefficient etc. '
             f'The roots are stored as [repeats, int(N/2)] arrays. ')

# <<<<<<<<<<<<<<<<<<< Calculations  >>>>>>>>>>>>>>>>>>
for i_N, N in enumerate(Ns):

    k_max = int(N / 2)  # maximum degree of matching polynomial, also number of roots

    if distrib == 'tilde_gaussian':
        As = generate_tildeX(N, repeats, mean=mean, stddev=stddev)
    elif distrib == 'gaussian':
        As = generate_X(N, repeats, mean=mean, stddev=stddev)
    elif distrib == 'sym_gaussian':
        As = generate_symX(N, repeats, mean=mean, stddev=stddev)
    else:
        raise ValueError(f'distribution {distrib} not recognized')


    np.save(DFUtils.create_filename(results_dir + fr'\raw\N={N}_tildeXs.npy'), As)

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

    np.save(DFUtils.create_filename(results_dir + fr'\raw\N={N}_mp_coeffs.npy'), coeffs)

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

    np.save(DFUtils.create_filename(results_dir + fr'\N={N}_roots.npy'), roots_N)

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
    save_name = results_dir + fr'\plots\N={N}.png'

    # Create a scatter plot in the complex plane
    roots_real = [np.real(r) for r in roots_dict[N]]
    roots_imag = [np.imag(r) for r in roots_dict[N]]

    plt.figure(f'N={N}')

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
    plt.scatter(roots_real, roots_imag, marker='x', s=10, label='Roots')

    # Set axes
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)


    plt.xscale('symlog', linthresh=linthresh)
    plt.yscale('symlog', linthresh=linthresh)

    plt.xlabel('Real')
    plt.ylabel('Imaginary')

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