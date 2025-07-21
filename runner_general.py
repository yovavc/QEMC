import glob
import os

from DbAdapter import DbAdapter

dbAdapter = DbAdapter()
graphs = glob.glob("regular_graphs/*")
layers = [50]
stepsizes = [0.14]
for stepsize in stepsizes:
    for graph in graphs:
        exploded = graph.split("_")
        numberOfNodes = int(exploded[5].split(".")[0])
        if 256 != numberOfNodes:
            continue

        for i in range(10):
            for layer in layers:

                ones = int(numberOfNodes / 2 + 1)

                for j in range(6):
                    ones = ones - 1
                    result = dbAdapter.get_current_experiments_count_expected_blacks(layer, stepsize, numberOfNodes, ones)
                    if not result or result[0]["count"] < 40:
                        print("python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize))
                        os.system("python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize))
