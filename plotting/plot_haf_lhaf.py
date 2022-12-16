import numpy as np
import matplotlib.pyplot as plt
from src.utils import DFUtils
import math


time_stamp = r'\24-11-2022(14-59-29.364186)'

M_min = 10
M_max = 25
Ms = list(range(M_min, M_max+1))
dir = r'..\Results\experiments' + time_stamp
plotdir = dir + r'\plots'

save_fig = False
if save_fig:
    test = DFUtils.create_filename(plotdir+r'\test.pdf')

dis_total_probs = np.zeros(M_max - M_min + 1)
undis_total_probs = np.zeros(M_max - M_min + 1)
for i, M in enumerate(Ms):
    dis_filename = DFUtils.return_filename_from_head(dir, r'dis_M={}'.format(M))
    undis_filename = DFUtils.return_filename_from_head(dir, r'undis_M={}'.format(M))

    ind1 = dis_filename.find('N=')
    N = int(dis_filename[ind1+2:-4])

    dis_probs = np.load(dis_filename)  # These are unnormalised |lhaf|^2
    dis_total_prob = sum(dis_probs)
    dis_total_probs[i] = dis_total_prob

    undis_probs = np.load(undis_filename)  # These are unnormalised |haf|^2
    undis_total_prob = sum(undis_probs)
    undis_total_probs[i] = undis_total_prob

    dis_probs_norm = dis_probs / dis_total_prob  # normalise
    dis_probs_norm[::-1].sort()
    num_probs = len(dis_probs_norm)

    undis_probs_norm = undis_probs / undis_total_prob # normalise
    undis_probs_norm[::-1].sort()


    plt.figure(f'normalised |lhaf|^2 for M={M}, N={N}')
    hist, bins = np.histogram(dis_probs_norm, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(dis_probs_norm, weights= np.ones_like(dis_probs_norm)/len(dis_probs_norm), alpha=0.5, bins=logbins, label='|lhaf|^2')

    hist2, bins2 = np.histogram(undis_probs_norm, bins=100)
    logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
    plt.hist(undis_probs_norm, weights=np.ones_like(undis_probs_norm) / len(undis_probs_norm), alpha=0.5, bins=logbins2, label='|haf|^2')

    plt.xscale('log')
    plt.xlabel('Normalised |lhaf|^2')
    plt.xlim(left=1e-10, right=1e0)
    plt.title('M={},N={}'.format(M,N))
    plt.legend()
    if save_fig:
        plt.savefig(plotdir+r'\hist_M={}_N={}.pdf'.format(M,N))


    plt.figure(f'|lhaf|^2 vs |haf|^2 for M={M}, N={N}')
    plt.plot(list(range(num_probs)), dis_probs_norm, label='|lhaf|^2')
    plt.plot(list(range(num_probs)), undis_probs_norm, label='|haf|^2')
    plt.yscale('log')
    plt.title('M={}, N={}'.format(M, N))
    plt.xticks([0, num_probs])
    plt.legend()
    if save_fig:
        plt.savefig(plotdir + r'\lhaf_vs_haf_M={}_N={}.pdf'.format(M, N))#

