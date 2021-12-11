import argparse
import networkx as nx
import max_cut_exhaustive_search_class

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
args = parser.parse_args()

graph = nx.read_graph6(args.fileName)
search = max_cut_exhaustive_search_class.mac_cut_exhaustive_search()

current_arcs = list(graph.edges)
group = search.get_max_cut_group(current_arcs)

print("graph:" + args.fileName +",GT:" + str(group) + ",group0Count:" + str(group.count("0"))
      + ",group1Count:" + str(group.count("1")))
