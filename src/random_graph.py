import numpy as np
import random


class RandomGraph:

    def __init__(self, M, max_degree):
        if max_degree > M:
            raise ValueError('Maximum degree must be attainable and cannot be larger than number of vertices')

        self.M = M
        self.max_degree = max_degree
        self.adj = self.__generate()

    def __generate(self):
        """Random unweighted loopless graph. """
        M = self.M
        max_degree = self.max_degree

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

    def adjacency_matrix(self):
        return self.adj

    def degrees(self):
        return np.sum(self.adj, 0)  # sum every column

    def degree_counts(self):
        count_dict = dict.fromkeys(range(self.max_degree+1), 0)
        for degree in self.degrees():
            count_dict[degree] += 1

        return count_dict