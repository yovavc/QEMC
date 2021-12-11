
import networkx as nx
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
args = parser.parse_args()

g = nx.read_graph6(args.fileName)

nx.draw(g)
plt.show()
