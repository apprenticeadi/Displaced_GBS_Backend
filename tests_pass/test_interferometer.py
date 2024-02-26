from src.gbs_experiment import PureGBS
from scipy.stats import unitary_group
from src.interferometer import Interferometer, square_decomposition

M = 11
U = unitary_group.rvs(M)

I = square_decomposition(U)
I.draw(show_params=False, show_plot=False)