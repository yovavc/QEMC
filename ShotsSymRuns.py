import glob
import math
import os
import random
import sys
from DbAdapter import DbAdapter
import networkx as nx
from natsort import natsorted, ns

metadata = {
    # 8: {"stepsize": 0.8, "layers": 1, "meanGW": 9.2, "maxGW": 10, "time": 0.1179 / 10},
    # 16: {"stepsize": 0.7, "layers": 5, "meanGW": 45.5, "maxGW": 46, "time": 0.1241 / 10},
    # 32: {"stepsize": 0.7, "layers": 5, "meanGW": 98.3, "maxGW": 102, "time": 0.1355 / 10},
    # 64: {"stepsize": 0.1, "layers": 50, "meanGW": 199.5, "maxGW": 204, "time": 0.2378 / 10},
    # 128: {"stepsize": 0.2, "layers": 40, "meanGW": 407.4, "maxGW": 415, "time": 0.5834 / 10},
    # 256: {"stepsize": 0.08, "layers": 80, "meanGW": 817.0, "maxGW": 830, "time": 5.3272 / 10},
    512: {"stepsize": 0.1, "layers": 70, "meanGW": 1633.8, "maxGW": 1672, "time": 110.3288 / 10},
    1024: {"stepsize": 0.12, "layers": 100, "meanGW": 3285.8, "maxGW": 3307, "time": 1966.1637 / 10},
    2048: {"stepsize": 0.08, "layers": 120, "meanGW": 6580.3, "maxGW": 6630, "time": 49234.7973 / 10},
}


def f0(val):
    return val


def f1(val):
    return val ** 2


def f2(val):
    return 2 * val ** 2


def f3(val):
    return 3 * val ** 2


def f4(val):
    return 4 * val ** 2


def f5(val):
    return 5 * val ** 2


shots_func = [f0, f1, f2, f3, f4, f5]
graphs = glob.glob("regular_graphs/*.g6")
graphs = natsorted(graphs, key=lambda y: y.lower())
dbAdapter = DbAdapter()
backend = "sym"
comment = "shots_vs_mean_gw_" + backend
for graph in graphs:
    g = nx.read_graph6(graph)
    # for graph in graphs:
    nodes = list(g.nodes)
    numberOfNodes = len(nodes)
    ones = int(numberOfNodes / 2)
    if numberOfNodes > 2048:
        continue
    if numberOfNodes in metadata:
        layer = metadata[numberOfNodes]["layers"]
        stepsize = metadata[numberOfNodes]["stepsize"]
        meanGW = metadata[numberOfNodes]["meanGW"]
        maxGW = metadata[numberOfNodes]["maxGW"]
        step = 2000
        for shot_func in shots_func:
            shots = shot_func(numberOfNodes)
            for i in range(10):
                # stepsize = random.choice(stepsizes)
                print("python3 with_ent_general_db_simulation.py " + str(ones) + " " + graph + " " + str(
                    layer) + " " + str(
                    stepsize) + " " + str(step) + " " + str(shots) + " " + backend + " " + comment + " " + str(
                    meanGW) + " " + str(maxGW))
                os.system("python3 with_ent_general_db_simulation.py " + str(ones) + " " + graph + " " + str(
                    layer) + " " + str(
                    stepsize) + " " + str(step) + " " + str(shots) + " " + backend + " " + comment + " " + str(
                    meanGW) + " " + str(maxGW))
