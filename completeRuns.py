import glob
import os
import random
import sys
from DbAdapter import DbAdapter

graphs = glob.glob("regular3_graphSmall/*")
graphs.reverse()

layers = [1, 2, 3]#5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
# layers = [100, 110, 120, 130, 140, 150, 200]
# stepsizes = [0.02 , 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
stepsizes = [0.6]
steps = [70]
dbAdapter = DbAdapter()
for step in steps:
    for i in range(sys.maxsize):
        graph = graphs[i % len(graphs)]
    # for graph in graphs:
        exploded = graph.split("_")
        numberOfNodes = int(exploded[5].split(".")[0])
        if numberOfNodes == 8192 :
            continue
        ones = int(numberOfNodes / 2)
        for layer in layers:
            # layer = random.choice(layers)
            for stepsize in stepsizes:
                # stepsize = random.choice(stepsizes)
                result = dbAdapter.get_current_experiments_count(layer, stepsize, numberOfNodes, step)
                print("python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize) + " " + str(step))
                if not result or result[0]["count"] < 5:
                    print(result)
                    # print( "python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize) + " " + str(step))
                    # os.system("python3 with_ent_general_db.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize) + " " + str(step))
                else:
                     print("OK")
