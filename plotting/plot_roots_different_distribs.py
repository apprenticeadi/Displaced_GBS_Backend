import time
from scipy.stats import gaussian_kde
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy.stats import bootstrap
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from src.utils import DFUtils

plt.show()
plt.ion()
matplotlib.use('TkAgg')

# tildeX
roots1 = np.load(r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2023-08-23(18-00-43.689530)\N=12_roots.npy')
# sym Gaussian
roots2 = np.load(r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2023-09-10(19-35-56.363960)\N=12_roots.npy')
# sub unitary
roots3 = np.load(r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2024-08-07(13-48-54.169288)\N=12_K=144_roots.npy')

N = 12
reps = 10000
fontsize = 18

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plot_dir = fr'..\Plots\roots_mp\different_distributions_{time_stamp}'

fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout='constrained', sharex='all', sharey='all')
bins = np.logspace(-7, 1, 150)
symbins = np.concatenate([-bins[::-1], bins])

distributions = ['tildeX', 'symGauss', 'subUnitary']
titles = ['(a)', '(b)', '(c)']
images = []
for i_roots, roots in enumerate([roots1, roots2, roots3]):

    roots = roots.reshape((reps, N//2))
    min_roots = np.zeros(reps, dtype=np.complex128)
    for i in range(reps):
        argmin = np.argmin(np.abs(roots[i]))
        min_roots[i] = roots[i, argmin]

    ax = axs[i_roots]

    x = np.real(min_roots)
    y = np.imag(min_roots)
    data, x_e, y_e = np.histogram2d(x, y, symbins)
    data = data / np.sum(data)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    image = ax.scatter(x, y, c=z, s=20, cmap='viridis', vmin=0, vmax=0.003)
    images.append(image)
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_yscale('symlog', linthresh=1e-3)

    spine_alpha = 0.5
    ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(spine_alpha)

    ax.axhline(0, color='black', linewidth=1, alpha=0.8)  # Real axis
    ax.axvline(0, color='black', linewidth=1, alpha=0.8)  # Imaginary axis

    ax.tick_params(axis='both', which='major', labelsize=fontsize-6)
    ax.set_xlabel(r'Re$(z)$', fontsize=fontsize - 2)
    ax.set_title(titles[i_roots], fontsize=fontsize, loc='left')
    np.savetxt(DFUtils.create_filename(plot_dir + rf'\{distributions[i_roots]}_N={N}_real_imag_density.txt'), np.vstack([x, y, z]).T)

fig.supylabel(r'Im$(z)$', fontsize=fontsize-2)
cbar = fig.colorbar(image, ax=axs, location='bottom', aspect=60, shrink=0.7)
cbar.ax.set_xlabel('Density of roots', fontsize=fontsize-2)
cbar.ax.tick_params(labelsize=fontsize-6)

fig.savefig(DFUtils.create_filename(plot_dir + fr'\roots_density.pdf'))
