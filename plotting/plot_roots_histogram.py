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

# dir1 = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2024-08-05(13-15-24.513015)'
# dir2 = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2023-08-25(17-08-36.941174)'
# dir1 = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2024-08-05(13-13-03.177595)'
# dir1 = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2024-08-07(13-48-54.169288)'
dir1 = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\roots_matching_polynomial\10000rep_2024-08-07(11-16-28.100724)'
reps = 10000

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
plot_dir = fr'..\Plots\roots_mp\symGauss_distrib_{time_stamp}'

fontsize = 18

fig = plt.figure(figsize=(8, 10), layout='constrained')
ax1 = fig.add_subplot(212)
Ns = [14, 12, 8, 4]  # [12, 10, 8, 6]
K = 16
bins = np.logspace(-7, 1, 200)
root_dict = {}
for N in Ns:
    # if N == K:
    #     roots = np.load(dir2+rf'\N=16_roots.npy')
    # else:
    roots = np.load(dir1+rf'\N={N}_K={K}_roots.npy')

    roots = roots.reshape((reps, N//2))
    min_roots = np.zeros(reps, dtype=np.complex128)
    for i in range(reps):
        argmin = np.argmin(np.abs(roots[i]))
        min_roots[i] = roots[i, argmin]

    root_dict[N] = min_roots

    ax1.hist(np.abs(min_roots), bins=bins, alpha=0.3, label=f'N={N}')

    # save
    np.savetxt(DFUtils.create_filename(plot_dir + fr'\N={N}_abs_min_roots.txt'), np.abs(min_roots))


ax1.set_xscale('log')
ax1.set_xlim([1e-3, 10])
ax1.set_xlabel(r'$|z|$', fontsize=fontsize-2)
ax1.set_ylabel(r'Roots per bin', fontsize=fontsize-2)
ax1.set_title(f'(b) K={K}', fontsize=fontsize, loc='left')
ax1.legend(fontsize=fontsize-2)
ax1.tick_params(axis='both', which='major', labelsize=fontsize-6)

roots = root_dict[Ns[0]]  # the minimum root of the 10,000 matching polynomials
x = np.real(roots)
y = np.imag(roots)

bins = np.logspace(-7, 1, 150)
symbins = np.concatenate([-bins[::-1], bins])

ax2 = fig.add_subplot(211)
data , x_e, y_e = np.histogram2d(x, y, symbins)
data = data / np.sum(data)
z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T ,
             method = "splinef2d", bounds_error = False)

#To be sure to plot all data
z[np.where(np.isnan(z))] = 0.0

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
image = ax2.scatter(x, y, c=z, s=20, cmap='viridis', )
ax2.set_xscale('symlog', linthresh=1e-3)
ax2.set_yscale('symlog', linthresh=1e-3)

np.savetxt(plot_dir + rf'\N={K}_real_imag_density.txt', np.vstack([x, y, z]).T)
# colorbar
# norm = Normalize(vmin = np.min(z), vmax = np.max(z))
cbar = fig.colorbar(image, ax=ax2, location='bottom', aspect=30)
cbar.ax.set_xlabel('Density of roots', fontsize=fontsize-2)
cbar.ax.tick_params(labelsize=fontsize-6)

ax2.set_title(f'(a) N={Ns[0]}, K={K}', fontsize=fontsize, loc='left')
ax2.set_xlabel(r'Re$(z)$', fontsize=fontsize-2)
ax2.set_ylabel(r'Im$(z)$', fontsize=fontsize-2)
ax2.tick_params(axis='both', which='major', labelsize=fontsize-6)

spine_alpha=0.5
for ax in [ax1, ax2]:
    ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(spine_alpha)
ax2.axhline(0, color='black', linewidth=1, alpha=0.8)  # Real axis
ax2.axvline(0, color='black', linewidth=1, alpha=0.8)  # Imaginary axis


# bootstrap
abs_roots = np.abs(roots)
# median
med_res = bootstrap((abs_roots,), np.median, confidence_level=0.95)
median = np.median(abs_roots)
med_bar = [median, median - med_res.confidence_interval.low, med_res.confidence_interval.high - median]
ax1.axvspan(median-med_bar[1], median+med_bar[2], color='black', alpha=1.)

# quarter
def find_quart(a, axis=0):
    return np.quantile(a, 0.25, axis=axis)

quart = find_quart(abs_roots)
quart_res = bootstrap((abs_roots,), find_quart, confidence_level=0.95)
quart_bar = [quart, quart - quart_res.confidence_interval.low, quart_res.confidence_interval.high - quart]
ax1.axvspan(quart-quart_bar[1], quart+quart_bar[2], color='red', alpha=1.)

# draw circle patches
circle_median = plt.Circle((0, 0), radius=median, color='black', fill=False, lw=2, label= r'50% non-zero') # label=rf'$|z|=${median:.3f}')
circle_quart = plt.Circle((0, 0), radius=quart, color='red', fill=False, lw=2, label=r'75% non-zero') # label=rf'$|z|=${quart:.3f}')
ax2.add_patch(circle_median)
ax2.add_patch(circle_quart)

# save
fig.savefig(DFUtils.create_filename(plot_dir + '\min_roots.pdf'))

df = pd.DataFrame(columns=['type', 'value', 'n_error', 'p_error'])
df.loc[0] = ['median', median, med_bar[1], med_bar[2]]
df.loc[1] = ['quart', quart, quart_bar[1], quart_bar[2]]
df.to_csv(plot_dir+r'\median_quart.csv', index=False)
