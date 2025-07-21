import os

import networkx as nx
import argparse
import matplotlib.pyplot as plt


files = os.listdir("graphs")
import csv
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for file in files:

        g = nx.read_graph6("graphs/" + file)

        graph = list(g.edges)
        spamwriter.writerow([file] + [graph])

        print(str(file) + " " + str(graph))
