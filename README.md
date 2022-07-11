# displaced_car_chase

Required packages in environment.yml file.(Some packages may be redundant). If it doesn't work, try environment_lin.yml.

Src folder gives the functions for generating symplectic matrices, and converting between Gaussian state and graphs. A few functions are directly borrowed from thewalrus, so need to add some disclaimers before making this public. I try to avoid using thewalrus package directly, because thewalrus works in a dimensional quadrature basis, where there is a hbar in the commutation relations, which is different from my notes. 

There is also a log_utils.py file that gives a logging function. 

A few scripts in 'script' folder mess around with certain degrees of freedom we have. 'worst_case_GBS.ipynb' is a jupyter notebook by Jake that tries to optimise for flat squeezing parameters. 



