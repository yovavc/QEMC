import glob
import os
import random
import sys
from DbAdapter import DbAdapter
import networkx as nx
from natsort import natsorted, ns

metadata = {
    # 4: {"stepsize": 0.9, "shots": 48, "layers": 1, "iterations": 10, "optimal_cut": 4, "meanGW": 3.7, "maxGW": 4},
    # 6: {"stepsize": 0.9, "shots": 108, "layers": 2, "iterations": 80, "optimal_cut": 7, "meanGW": 7, "maxGW": 7},
    # 8: {"stepsize": 0.9, "shots": 192, "layers": 2, "iterations": 250, "optimal_cut": 10, "meanGW": 10, "maxGW": 10},
    # 10: {"stepsize": 0.95, "shots": 300, "layers": 2, "iterations": 100, "optimal_cut": 12, "meanGW": 11, "maxGW": 12},
    # 12: {"stepsize": 0.98, "shots": 432, "layers": 3, "iterations": 300, "optimal_cut": 16, "meanGW": 15.1, "maxGW": 16},
    # 14: {"stepsize": 0.99, "shots": 588, "layers": 3, "iterations": 325, "optimal_cut": 19, "meanGW": 18.6, "maxGW": 19},
    # 16: {"stepsize": 0.95, "shots": 768, "layers": 5, "iterations": 350, "optimal_cut": 21, "meanGW": 20.3, "maxGW": 21},
    16: {"stepsize": 0.7, "shots": 768, "layers": 2, "iterations": 350, "optimal_cut": 21, "meanGW": 20.3,
         "maxGW": 21},
    # 32: {"stepsize": 0.95, "shots": 768, "layers": 5, "iterations": 1000, "optimal_cut": 42, "meanGW": 20.3, "maxGW": 21}
}

graphs = glob.glob("google_graphs/*.g6")
graphs = natsorted(graphs, key=lambda y: y.lower())
dbAdapter = DbAdapter()
backend = "ibmq_jakarta"
comment = "HW_run_on_" + backend
for graph in graphs:
    g = nx.read_graph6(graph)
    # for graph in graphs:
    nodes = list(g.nodes)
    numberOfNodes = len(nodes)
    ones = int(numberOfNodes / 2)
    if numberOfNodes in metadata:
        layer = metadata[numberOfNodes]["layers"]
        stepsize = metadata[numberOfNodes]["stepsize"]
        step = metadata[numberOfNodes]["iterations"]
        shots = metadata[numberOfNodes]["shots"]
        optimal_cut = metadata[numberOfNodes]["optimal_cut"]
        for i in range(10):
        # stepsize = random.choice(stepsizes)
            str_command = "python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize) + " " + str(step) + " " + str(shots) + " " + backend + " " + comment + " " + str(optimal_cut)
            print(str_command)
            os.system(str_command)


