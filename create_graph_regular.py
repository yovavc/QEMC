from networkx.generators.random_graphs import random_regular_graph
import networkx as nx
sizes = [3072]
regular = 9
for entry in sizes:
    size = entry
    graph = random_regular_graph(regular, size)
    string = "regular9_graph4/vertices_regular_" + str(regular) + "_size_" + str(size) + ".g6"
    nx.write_graph6(graph, string)
