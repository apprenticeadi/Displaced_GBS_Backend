import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils import DFUtils

x = -1
r_max = 1

if x < 0:
    complexity = 'hard'
else:
    complexity = 'easy'
M_list = [5, 8, 10, 13, 15, 18, 20]
color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
color_dict = dict(zip(M_list, color_list))

M_dict = {}

for M in M_list:

    data_df, data_name = DFUtils.read_filename_head(
        directory= r'..\Results\probe_even_v\{}'.format(complexity),
        filename_head= r'M={}'.format(M)
    )
    M_dict[M] = (data_df['log_diag_weight'], data_df['sq_photons'], data_df['coh_photons'])

stamp_idx = data_name.rfind('_') + 1
timestamp = data_name[stamp_idx:]

plt.figure(0)
for M in M_dict:
    tup = M_dict[M]
    plt.plot(tup[0], tup[1] / tup[2], color=color_dict[M], linestyle='-', label='M={}'.format(M))
plt.xlabel('log diagonal weight')
plt.xticks(tup[0])
plt.ylabel('squeezed photons / classical photons')
plt.yscale('log')
plt.ylim([0.01, 1000])
plt.legend()
plt.title('x={}, sq/cl'.format(x))
plt.savefig(DFUtils.create_filename(r'..\Plots\probe_even_v\{}_sq_vs_cl_{}.pdf'.format(complexity, timestamp)))

plt.figure(1)
for M in M_dict:
    tup = M_dict[M]
    plt.plot(tup[0], tup[2], color=color_dict[M], linestyle='-', label='M={}'.format(M))
plt.xlabel('log diagonal weight')
plt.xticks(tup[0])
plt.ylabel('classical photons')
plt.yscale('log')
plt.ylim([0.001, 100])
plt.legend()
plt.title('x={}, cl photons'.format(x))
plt.savefig(DFUtils.create_filename(r'..\Plots\probe_even_v\{}_cl_photons_{}.pdf'.format(complexity, timestamp)))

plt.figure(2)
for M in M_dict:
    tup = M_dict[M]
    plt.plot(tup[0], tup[1], color=color_dict[M], linestyle='-', label='M={}'.format(M))
    plt.plot(tup[0], [M * np.sinh(r_max)**2] * len(tup[0]), color=color_dict[M], linestyle='--')
plt.xlabel('log diagonal weight')
plt.xticks(tup[0])
plt.ylabel('sq photons')
plt.legend()
plt.title('x={}, sq photons'.format(x))
plt.savefig(DFUtils.create_filename(r'..\Plots\probe_even_v\{}_sq_photons_{}.pdf'.format(complexity, timestamp)))