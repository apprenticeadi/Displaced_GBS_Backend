import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numba
import logging
from scipy import stats
from scipy.special import comb

from src.utils import DGBSUtils
from src.photon_number_distributions import total_displaced_squeezed_vacuum, vac_prob_displaced_squeezed_vacuum, big_F

""" Script to calculate and plot total photon number probability p(N) of getting N photons"""

Ns = np.arange(start=2, stop=101)
w = 1
r, beta = DGBSUtils.solve_w(w, 1)  # because N=K so we want 1 mean photon number per mode
print((r,beta))
assert np.isclose(DGBSUtils.calc_w(r, beta), w)

pNs = np.zeros_like(Ns, dtype=float)
p0s = np.zeros_like(Ns, dtype=float)
bigFs = np.zeros_like(Ns, dtype=float)
for i, N in enumerate(Ns):

    K = N
    rs = r * np.ones(K, dtype=float)
    betas = beta * np.ones(K, dtype=float)

    t0 = time.time()
    total_probs = total_displaced_squeezed_vacuum(rs, betas, N)
    t1 = time.time()

    p0 = vac_prob_displaced_squeezed_vacuum(rs, betas)

    print(f'N={N}, pN={total_probs[N]} in time={t1-t0}')

    pNs[i] = total_probs[N]
    p0s[i] = p0
    bigFs[i] = big_F(w, N, N)[N]


Ns = np.asarray(Ns, dtype=float)

plt.figure('pN')
plt.plot(Ns, p0s, label='p0')
plt.plot(Ns, pNs, label='pN')
plt.plot(Ns, 1/Ns**2, label='1/N^2')
plt.plot(Ns, 1/np.exp(Ns), label='exp(-N)')
plt.plot(Ns, np.power(np.tanh(r), Ns), label='tanh(r)^N')
plt.yscale('log')
plt.legend()

plt.figure('p0/pN')
plt.plot(Ns, np.power(np.tanh(r), Ns) * p0s / pNs, label='tanh(r)^N * p0/pN')
plt.plot(Ns, np.power(2, Ns) / bigFs, label='2^N/F(w)')
plt.plot(Ns, 1/bigFs, label='1/F(w)')
plt.plot(Ns, 1/comb(Ns**2 + Ns - 1, Ns), label='(N^2+N-1 choose N)')
plt.plot(Ns, 1/comb(Ns**2, Ns), label='(N^2 choose N)')
plt.yscale('log')
plt.legend()

log_factor = np.log(np.power(2, Ns) / bigFs)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ns, log_factor)
plt.figure('log factor')
plt.plot(Ns, log_factor, 'x', label='ln(2^N/F(w))')
plt.plot(Ns, slope * Ns + intercept, label='linear regression')
plt.legend()
# min_ylim, max_ylim = plt.ylim()
# min_xlim, max_ylim = plt.xlim()
plt.text(40 ,-40, f'y={slope:.3f}N{intercept:+.3f}')