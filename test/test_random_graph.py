from src.random_graph import RandomGraph

test_graph = RandomGraph(5, 3)

adj = test_graph.adjacency_matrix()

print(adj)
print(test_graph.degrees())
print(test_graph.degree_counts())
