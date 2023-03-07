import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from src.utils import MatrixUtils
# <<<<<<<<<<<<<<<<<<< Size  >>>>>>>>>>>>>>>>>>
M = 100
N = int(np.floor(np.sqrt(M)))
w = 1   # The diagonal weight.

U = unitary_group.rvs(M)

B = U @ U.T  # The tanhr term is absorbed inside w
half_gamma = w * np.sum(U, axis=1)
x = B / np.outer(half_gamma, half_gamma)
x = MatrixUtils.filldiag(x, np.zeros(M))  # We don't want x_ii
masked_x = x[
            ~np.eye(len(x), dtype=bool)]  # mask out diagonal terms which are zero. masked shaped is 1d
x_n = x[:N, :N]
abs_x = np.absolute(x_n)
sum_x = np.sum(abs_x, axis=1)

axis_lim = 1e3

plt.figure(f'Plot of |x| for M={M}, w={w}')
plt.axhline(y=0, xmin=-axis_lim, xmax=axis_lim, color='black')
plt.axvline(x=0, ymin=-axis_lim, ymax=axis_lim, color='black')
circle1 = plt.Circle((0,0), 1, color='r', alpha=0.5, label='Radius 1')
circle2 = plt.Circle((0,0), 1/N, color='yellow', alpha=1, label='Radius 1/N')
circle3 = plt.Circle((0,0), 1/N**4, color='orange', alpha=1, label='Radius 1/N**4')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)
plt.scatter(masked_x.real, masked_x.imag, marker='.')
plt.xlim(left=-axis_lim, right=axis_lim)
plt.ylim(bottom=-axis_lim, top=axis_lim)
plt.yscale('symlog', linthresh = 10**np.floor(np.log10(1/N**4)))
plt.xscale('symlog', linthresh = 10**np.floor(np.log10(1/N**4)))
plt.legend()
plt.title(f'Scatter plot of x_ij for M={M}, w={w}')

plt.figure(f'Plot of sum |x| for M={M}, w={w}')
plt.axhline(y=0, xmin=-axis_lim, xmax=axis_lim, color='black')
plt.axvline(x=0, ymin=-axis_lim, ymax=axis_lim, color='black')
circle1 = plt.Circle((0,0), 1, color='r', alpha=0.5, label='Radius 1')
circle2 = plt.Circle((0,0), 1/N, color='yellow', alpha=1, label='Radius 1/N')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.scatter(sum_x.real, sum_x.imag, marker='.')
plt.xlim(left=-axis_lim, right=axis_lim)
plt.ylim(bottom=-axis_lim, top=axis_lim)
plt.yscale('symlog', linthresh = 10**np.floor(np.log10(1/N)))
plt.xscale('symlog', linthresh = 10**np.floor(np.log10(1/N)))
plt.legend()
plt.title(f'Scatter plot of sum x_ij for M={M}, w={w}')



def func(r):
    if np.sinh(r) ** 2 >= 1 / np.sqrt(M) or r <= 0:
        return 100000
    else:
        return np.sqrt(1 / np.sqrt(M) - np.sinh(r)**2) * (1 - np.tanh(r)) / np.sqrt(np.tanh(r)) - w

root = fsolve(func, np.arcsinh(np.sqrt( 0.1 / np.sqrt(M))))
r = root[0]
beta = np.sqrt(1 / np.sqrt(M) - np.sinh(r)**2)

print(f'r={r}')
print(f'beta={beta}')

N_sq = np.sinh(r) ** 2
N_dis = beta ** 2

sq_dis_ratio = N_sq / N_dis

print(sq_dis_ratio)