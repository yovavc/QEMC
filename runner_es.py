import glob
import os

graphs = glob.glob("regular3_graphs/*")
graphs.sort()
for graph in graphs:

    os.system("python3 max_cut_exhaustive_search.py " + graph + " > graph_gt/" + os.path.basename(graph) + ".maxcut.txt")
