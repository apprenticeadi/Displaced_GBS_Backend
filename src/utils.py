import os
import pandas as pd
import random
import numpy as np
import sys
import logging
import datetime
import copy
from scipy.optimize import fsolve
import time

import src.interferometer as itf


class MatrixUtils:

    @staticmethod
    def remove_small(a, tol=1e-7):
        """
        :param a: Any np array or list
        :param tol: Relative tolerance compared to the average
        :return: a with small elements removed
        """
        anew = copy.deepcopy(np.asarray(a, dtype=np.complex64))
        a_flat = anew.flatten()
        av = a_flat.sum() / a_flat.shape[0]

        anew.real[abs(anew.real) / abs(av) < tol] = 0.0
        anew.imag[abs(anew.imag) / abs(av) <tol] = 0.0

        return anew

    @staticmethod
    def filldiag(A, gamma):
        A=np.asarray(A)
        gamma = np.asarray(gamma)

        m,n = A.shape
        if m != n:
            raise ValueError('Input matrix should be square matrix')
        if m != gamma.shape[0]:
            raise ValueError('Input matrix and vector should have compatible shapes')

        return A + (gamma - A.diagonal()) * np.eye(m)

    @staticmethod
    def n_repetition(A, n_vector):
        A = np.asarray(A)
        n_vector = np.asarray(n_vector)
        if A.shape[0] != n_vector.shape[0]:
            raise ValueError('Input arrays should have compatible shapes')

        if len(A.shape) == 1:
            # This is vector
            A = np.repeat(A, repeats=n_vector, axis=0)
        elif len(A.shape) ==2:
            # This is matrix
            m,n = A.shape
            if m != n:
                raise ValueError('Input matrix should be square matrix')
            A = np.repeat(A, repeats=n_vector, axis=0)
            A = np.repeat(A, repeats=n_vector, axis=1)
        else:
            raise ValueError('Can only n-repeat matrix or vectors')

        return A


class DFUtils:

    @staticmethod
    def create_filename(filename: str):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename

    @staticmethod
    def read_filename_head(directory, filename_head, idx=0, dt=None):
        file_to_read = DFUtils.return_filename_from_head(directory, filename_head, idx)

        df = pd.read_csv(
            file_to_read,
            dtype=dt,
        )

        return df, file_to_read

    # Not necessarily pd dataframe
    @staticmethod
    def return_filename_from_head(directory, filename_head, idx=0):
        files = os.listdir(directory)

        filtered_files = [file_ for file_ in files if file_.startswith(filename_head)]
        file_to_read = os.path.join(directory, filtered_files[idx])
        return file_to_read


class RandomUtils:

    @staticmethod
    def random_adj(M, max_degree):
        """
        Generate adjacency matrix for random unweighted loopless graph. No restrictions on the possible edges a vertex can have.

        :param M: Number of vertices
        :param max_degree: Maximum degree of the graph

        :return: Adj: Adjacency matrix
        """

        adj = np.zeros((M, M), dtype=int)

        unfilled_vertices = list(range((M)))
        random.shuffle(unfilled_vertices)

        for i, start_vertex in enumerate(unfilled_vertices):
            end_vertices = unfilled_vertices.copy()
            end_vertices.remove(start_vertex)
            start_vertex_degree = adj[start_vertex, :].sum()

            if i == 0:
                # Force 0th vertex to attain max degree, so that the max degree is at least attained once
                num_neighbour = max_degree
            else:
                num_neighbour = np.random.randint(0, min(max_degree + 1 - start_vertex_degree, M - i))
            possible_edge_ends = np.random.choice(end_vertices, num_neighbour, replace=False)

            for possible_edge_end in possible_edge_ends:
                adj[start_vertex, possible_edge_end] = 1
                adj[possible_edge_end, start_vertex] = 1

                end_vertex_degree = adj[possible_edge_end, :].sum()
                assert end_vertex_degree <= max_degree
                if end_vertex_degree == max_degree:
                    unfilled_vertices.remove(possible_edge_end)

            if start_vertex_degree + num_neighbour == max_degree:
                unfilled_vertices.remove(start_vertex)

        return adj

    @staticmethod
    def random_interferometer(M, depth):
        """
        Generates random interferometer in Clements scheme, where angles for every phase shifter and beamsplitter
        given by a random number between 0 and 2pi
        :param M: Number of modes
        :param depth: Depth of interferometer

        :return: Interferometer object
        """

        I = itf.Interferometer()

        for k in range(depth):
            p = M // 2
            q = M % 2
            if k % 2 != 0 and q == 0:
                shift = 1
            else:
                shift = 0

            for i in range(p - shift):
                j = 2 * i + 1 + k % 2  # Clements interferometer mode index starts from 1
                phase = 2 * random.random() * np.pi
                angle = 2 * random.random() * np.pi

                bs = itf.Beamsplitter(j, j + 1, angle, phase)

                I.add_BS(bs)

        return I


class LogUtils:

    @staticmethod
    def log_config(time_stamp, dir=None, filehead='', module_name='', level=logging.INFO):
        # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
        if dir is None:
            dir = r'..\Results\logs'
        logging_filename = dir + r'\{}_{}.txt'.format(filehead, time_stamp)
        os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

        stdout_handler = logging.StreamHandler(sys.stdout)

        logging.basicConfig(filename=logging_filename, level=level,
                            format='%(levelname)s %(asctime)s %(message)s')

        # make logger print to console (it will not if multithreaded)
        logging.getLogger(module_name).addHandler(stdout_handler)


class TestUtils:

    @staticmethod
    def sf_circuit(M, U, alphas, rs, order='SUD', hbar=1):
        """Generate strawberryfields state for GBS circuit from strawberryfields"""

        import strawberryfields as sf
        from strawberryfields import ops

        sf.hbar = hbar

        eng = sf.Engine(backend='gaussian')
        prog = sf.Program(M)  # creates an M mode quantum program

        U = U.conjugate()  # Conjugate here due to convention difference
        rs = - np.atleast_1d(rs)  # Add minus sign here, because strawberry fields use different convention for squeezing (off by minus sign)
        alphas = np.atleast_1d(alphas)

        with prog.context as q:

            for operation in list(order):
                if operation == 'S':
                    for i, r_i in enumerate(rs):
                        ops.Sgate(r=np.absolute(r_i), phi=np.angle(r_i)) | q[i]
                elif operation == 'U':
                    ops.Interferometer(U) | q
                elif operation == 'D':
                    for j, alpha_j in enumerate(alphas):
                        ops.Dgate(r=np.absolute(alpha_j), phi=np.angle(alpha_j)) | q[j]
                else:
                    raise ValueError('Order not recognized')

        state = eng.run(prog).state
        return state


class DGBSUtils:
    """
    Some util functions for DisplacedGBS
    """

    @staticmethod
    def calc_w(r, beta):
        return (beta.conjugate() - beta * np.tanh(r)) / np.sqrt(np.tanh(r))

    @staticmethod
    def solve_w(w, N_mean, guess_r = 0):
        """
        Find real squeezing and displacement parameters that satisfy
        w = beta * (1-tanh(r)) / sqrt(tanh(r))
        and
        beta^2 + sinh(r)^2 = N_mean

        :return: real squeezing and displacement parameters (r,beta)
        """

        def cost(r):
            if np.sinh(r) ** 2 >= N_mean or r <= 0:
                return 100000
            else:
                return np.sqrt(N_mean - np.sinh(r) ** 2) * (1 - np.tanh(r)) / np.sqrt(np.tanh(r)) - w

        if guess_r == 0:
            if w <= 2:
                guess_r = 0.2
            elif w <= 3.5:
                guess_r = 0.1
            elif w <= 6:
                guess_r = 0.03
            elif w <= 10:
                guess_r = 0.01
            elif w <= 30:
                guess_r = 0.001
            else:
                guess_r = 0.0001

        root = fsolve(cost, guess_r)
        r = root[0]
        beta = np.sqrt(N_mean - np.sinh(r) ** 2)

        return r, beta

    @staticmethod
    def read_w_label(w_label, N):
        """Read labels such as 'w=1', 'w=2.3N^0.25' etc. """
        if 'N' in w_label:

            if w_label.index('=') + 1 == w_label.index('N'):
                w_scale=1
            else:
                w_scale = float(w_label[w_label.index('=') + 1: w_label.index('N')])

            w_exp = float(w_label[w_label.index('N') + 2:])
            w = w_scale * N ** w_exp
        else:
            w = float(w_label[2:])

        return w


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """Function for progress bar. https://stackoverflow.com/questions/3160699/python-progress-bar """

    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)
