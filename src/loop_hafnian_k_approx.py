import numpy as np
from thewalrus import hafnian
import itertools



# TODO: adapt the method to take in B matrix as well
def loop_hafnian_approx_batch(A_n, D_n, approx=2):
    """
    A clever batching method that calculates up to the k-th order approximation while greatly reducing
    the number of hafnian computations

    """

    # fix A and D to be 128 bit complex numbers
    A_n = np.asarray(A_n, dtype=np.complex128)
    D_n = np.asarray(D_n, dtype=np.complex128)

    N2 = len(D_n)  # Twice the number of photons

    if approx % 2 == 1:  # if the appoximation order is odd, it does not improve on the even order below it
        approx -= 1  # so might as well reduce it
    if approx == 0:
        return np.prod(D_n)  # 0th order is just the product of D
    if approx >= N2:
        # appox order >N is meaningless
        return hafnian(A_n + (D_n - A_n.diagonal()) * np.eye(N2), loop=True)  # _calc_loop_hafnian(A_n, D_n, np.ones(N // 2, dtype=np.int64), glynn=glynn)
    else:
        H = 0
        for output in itertools.combinations(range(N2), N2 - approx):
            # takes all choices of (N-approx) indices to be fixed into loops
            loops = np.asarray(output, dtype=int)

            # make array that is 0 for every index fixed into a loop, 1 for others
            reps = np.ones(N2, dtype=int)
            reps[loops] = 0

            # make a temporary version of D
            # only copy the values that come after the last entry in 'loops'
            # this avoids some double counting
            Dnew = np.zeros(N2, dtype=np.complex128)
            Dnew[loops[-1] + 1:] = D_n[loops[-1] + 1:] # this line wouldn't work for approx=N2

            # take submatrices - only keep indices which aren't fixed into loops
            Ds = Dnew[reps == 1]
            As = A_n[reps == 1, :]
            As = As[:, reps == 1]

            # add the product of D for the indices fixed in loops
            # times the loop hafnian of those that aren't
            # loop hafnian function could be replaced with something from thewalrus
            haf_term = hafnian(As + (Ds - As.diagonal())*np.eye(As.shape[0]), loop=True)
            gamma_prod = np.prod(D_n[loops])
            H_term = gamma_prod * haf_term

            # print('k={},output={},gamma_prod={},haf={}'.format(approx / 2, output, gamma_prod, haf_term))

            H += H_term
            # _calc_loop_hafnian(As, Ds, np.ones(approx // 2, dtype=np.int64), glynn=glynn)
        return H
