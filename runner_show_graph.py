import glob
import os

graphs = glob.glob("regular_graphs/*")
graphs.sort()

for graph in graphs:
    os.system("python3 show_graph.py " + graph + " graph_draw/")
