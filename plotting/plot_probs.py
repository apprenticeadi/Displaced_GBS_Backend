import numpy as np
import matplotlib.pyplot as plt
from src.utils import DFUtils
import math


time_stamp = r'\02-11-2022(18-33-30.045785)'

M_min = 10
M_max = 24
Ms = list(range(M_min, M_max+1))
dir = r'..\Results\experiments' + time_stamp
plotdir = dir + r'\plots'

save_fig = True
if save_fig:
    test = DFUtils.create_filename(plotdir+r'\test.pdf')

total_probs = np.zeros(M_max - M_min + 1)
sec_moments = np.zeros(M_max - M_min + 1)
for i, M in enumerate(Ms):
    filename = DFUtils.return_filename_from_head(dir, r'M={}'.format(M))

    ind1 = filename.find('N=')
    N = int(filename[ind1+2:-4])

    probs = np.load(filename)
    total_prob = sum(probs)
    total_probs[i] = total_prob

    probs_norm = probs / total_prob
    probs_norm[::-1].sort()
    sec_moments[i] = np.average(probs_norm**2)

    plt.figure(i+1)
    hist, bins = np.histogram(probs_norm, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(probs_norm, weights= np.ones_like(probs_norm)/len(probs_norm), bins=logbins)
    plt.xscale('log')
    plt.xlabel('Outcome probabilities')
    plt.xlim(left=1e-10, right=1e0)
    plt.title('M={},N={}'.format(M,N))
    if save_fig:
        plt.savefig(plotdir+r'\hist_M={}_N={}.pdf'.format(M,N))


plt.figure('Total prob scaling')
plt.plot(Ms, total_probs)
plt.xticks(Ms)
plt.xlabel('Mode number M')
plt.ylabel('Probability')
plt.title('Probability of collisionless N-photon coincidence for N=M^0.5')
if save_fig:
    plt.savefig(plotdir+r'\collisionless_prob_{}-{}modes.pdf'.format(M_min, M_max))

plt.figure('Second moment scaling')
plt.plot(Ms, sec_moments)
plt.xticks(Ms)
plt.xlabel('Mode number M')
plt.ylabel('E[p^2]')
plt.yscale('log')
plt.title('Second moment for N=M^0.5')
if save_fig:
    plt.savefig(plotdir+r'\second_moments_{}-{}modes.pdf'.format(M_min, M_max))