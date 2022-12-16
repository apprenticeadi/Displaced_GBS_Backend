import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
from sympy.utilities.iterables import multiset_permutations
import math
import time

from src.gbs_experiment import PureGBS, sudGBS, sduGBS
from src.utils import LogUtils

M = 10
k = 0.5  # total mean photon number = M^k

r = np.arcsinh(np.sqrt(M ** (k-1)))  # identical squeezing across input modes

