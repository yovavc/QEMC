import glob
import os
import subprocess
from natsort import natsorted, ns
import networkx as nx
graphs = glob.glob("graphs/vertices_32_*")
graphs = natsorted(graphs, key=lambda y: y.lower())

for graph in graphs:
    g = nx.read_graph6(graph)
    nodes = list(g.nodes)
    for i in range(1):
        for t in range(1, 1 + int(len(nodes)/2)):
            os.system("python with_ent_general.py "
                      + str(t)
                      + " "
                      + graph
                      + " > with_ent_general_up_to_div_2_result/"
                      + os.path.basename(graph)
                      + "_"
                      + str(i)
                      + "_b_"
                      + str(t)
                      + ".txt")
            os.system("python no_ent_general.py "
                      + str(t)
                      + " "
                      + graph
                      + " > no_ent_general_up_to_div_2_result/"
                      + os.path.basename(graph)
                      + "_"
                      + str(i)
                      + "_b_"
                      + str(t)
                      + ".txt")

