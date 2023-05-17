import numpy as np
from thewalrus import hafnian
import itertools


def loop_hafnian_approx_batch(B, gamma, k=1):
    """
    A clever batching method that calculates up to the k-th order approximation while greatly reducing
    the number of hafnian computations.
    The method should work for any general loop Hafnian, regardless of whether it is A or B matrix.
    """

    # fix A and D to be 128 bit complex numbers
    B = np.asarray(B, dtype=np.complex128)
    gamma = np.asarray(gamma, dtype=np.complex128)

    N = len(gamma)  # Number of photons for B matrix (allowed to be odd), or twice the number of photons for A matrix

    if k == 0:
        return np.prod(gamma)  # 0th order is just the product of D
    if k >= N//2:
        # k can only be between 0 and N//2
        return hafnian(B + (gamma - B.diagonal()) * np.eye(N), loop=True)  # _calc_loop_hafnian(A_n, D_n, np.ones(N // 2, dtype=np.int64), glynn=glynn)
    else:
        H = 0
        for output in itertools.combinations(range(N), N - 2*k):
            # takes all choices of (N-approx) indices to be fixed into loops
            loops = np.asarray(output, dtype=int)

            # make array that is 0 for every index fixed into a loop, 1 for others
            reps = np.ones(N, dtype=int)
            reps[loops] = 0

            # make a temporary version of D
            # only copy the values that come after the last entry in 'loops'
            # this avoids some double counting
            gamma_new = np.zeros(N, dtype=np.complex128)
            gamma_new[loops[-1] + 1:] = gamma[loops[-1] + 1:] # this line wouldn't work for approx=N2

            # take submatrices - only keep indices which aren't fixed into loops
            gamma_phi = gamma_new[reps == 1]
            B_phi = B[reps == 1, :]
            B_phi = B_phi[:, reps == 1]

            # add the product of D for the indices fixed in loops
            # times the loop hafnian of those that aren't
            # loop hafnian function could be replaced with something from thewalrus
            haf_term = hafnian(B_phi + np.diag(gamma_phi - B_phi.diagonal()), loop=True)
            gamma_prod = np.prod(gamma[loops])
            H_term = gamma_prod * haf_term

            # print('k={},output={},gamma_prod={},haf={}'.format(approx / 2, output, gamma_prod, haf_term))

            H += H_term
            # _calc_loop_hafnian(As, Ds, np.ones(approx // 2, dtype=np.int64), glynn=glynn)
        return H
