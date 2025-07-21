import os

import networkx as nx
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
# parser.add_argument("plotDir", type=str)
args = parser.parse_args()

g = nx.read_graph6(args.fileName)
maxColor = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]

nx.draw(g, with_labels=True , node_color=maxColor)

# plt.savefig(args.plotDir + os.path.basename(args.fileName) + ".png")
plt.show()
