import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import bootstrap

from src.utils import DFUtils, DGBSUtils


def get_cdf(data, x, bootstrapping=False, confidence_level=0.9):
    """cumulative distribution function"""

    data_converted = data < x
    cdf = np.mean(data_converted)

    if bootstrapping:
        print('Start bootstrapping')
        t1 = time.time()
        bootstrap_res = bootstrap((data_converted,), np.mean, confidence_level=confidence_level, batch=100,
                                  method='basic')
        t2 = time.time()
        # print(f'end bootstrapping after {t2-t1}s')
        n_error = cdf - bootstrap_res.confidence_interval.low
        p_error = bootstrap_res.confidence_interval.high - cdf
    else:
        n_error = 0
        p_error = 0

    return cdf, n_error, p_error


plt_dir = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Plots\acc_numerics\2024-08-08(10-46-44.108248)'

results_dir = r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Results\anticoncentration_over_X\100000repeats\prod_06-09-2024(10-09-21.994485)'
epsilons = np.load(
    r'C:\Users\zl4821\PycharmProjects\displaced_car_chase\Plots\acc_numerics\2024-08-08(10-46-44.108248)\data\epsilons.npy')

for N in [6, 10, 16, 20, 28, 34]:

    raw = np.load(results_dir + rf'\N={N}\raw_0.npy')

    cdfs = np.zeros((len(epsilons)), dtype=float)
    error_bars = np.zeros((2, len(epsilons)), dtype=float)

    for i_epsilon, epsilon in enumerate(epsilons):
        print(f'N={N}, epsilon={epsilon}')
        t1 = time.time()
        cdf, n_error, p_error = get_cdf(raw, 1 / epsilon, bootstrapping=True, confidence_level=0.95)
        t2 = time.time()
        print(f'Calculate finished after {t2 - t1}s. Results={(cdf, n_error, p_error)}')

        cdfs[i_epsilon] = cdf
        error_bars[:, i_epsilon] = np.array([n_error, p_error])

    np.save(DFUtils.create_filename(plt_dir + rf'\data\prod\N={N}_cdfs.npy'), cdfs)
    np.save(DFUtils.create_filename(plt_dir + rf'\data\prod\N={N}_errors.npy'), error_bars)

''' Bootstrap median/first quartile  '''


def find_quart(a, axis=0):
    return np.quantile(a, 0.25, axis=axis)


df = pd.DataFrame(
    columns=['N', 'median', 'median_n_error', 'median_p_error', 'quart', 'quart_n_error', 'quart_p_error'])

for i_N, N in enumerate(np.arange(6, 35)):
    raw = np.load(results_dir + rf'\N={N}\raw_0.npy')
    re_median = np.median(raw)
    re_quart = find_quart(raw)

    print(f'Start bootstrapping mean for N={N}')
    t1 = time.time()
    mean_boot = bootstrap((raw,), np.median, confidence_level=0.95, batch=100, method='basic')
    t2 = time.time()
    print(f'End bootstrapping median for N={N} after {t2 - t1}s')

    print(f'Start bootstrapping quartile for N={N}')
    t1 = time.time()
    quart_boot = bootstrap((raw,), find_quart, confidence_level=0.95, batch=100, method='basic')
    t2 = time.time()
    print(f'End bootstrapping quartile for N={N} after {t2 - t1}s')

    df.loc[i_N] = [N, re_median, re_median - mean_boot.confidence_interval.low,
                   mean_boot.confidence_interval.high - re_median,
                   re_quart, re_quart - quart_boot.confidence_interval.low,
                   quart_boot.confidence_interval.high - re_quart]


df.to_csv(DFUtils.create_filename(plt_dir + rf'\data\prod\bootstrapping_df.csv'))

