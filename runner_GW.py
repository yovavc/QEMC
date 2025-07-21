import glob
import os
from re import compile, split
dre = compile(r'(\d+)')
graphs = glob.glob("regular9_graph4/*")
graphs.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in split(dre, l)])
# graphs.sort()
num_trials = 1
for graph in graphs:
    print("python3 Adi_GW.py " + graph + " > regular9_graph4_GW/" + os.path.basename(graph) + ".gw_" +".txt " + str(num_trials))
    os.system("python3 Adi_GW.py " + graph + " > regular9_graph4_GW/" + os.path.basename(graph) + ".gw_" +".txt " + str(num_trials))

