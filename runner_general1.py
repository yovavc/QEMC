import glob
import os


graphs = glob.glob("regular_graphs/*")
graphs.reverse()
layers = [20]
stepsizes = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
for graph in graphs:
    exploded = graph.split("_")

    for i in range(10):
        for layer in layers:
            ones = int(int(exploded[5].split(".")[0]) / 2)
            for stepsize in stepsizes:
                os.system(
                    "python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize))
