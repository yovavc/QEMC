import glob
import os


graphs = glob.glob("regular_graphs2/*")
graphs.reverse()
layers = [20]
stepsizes = [0.02]
beta1s = [0.8, 0.81, 0.83, 0.85, 0.87, 0.89, 0.895, 0.899]
beta2s = [0.99, 0.991, 0.993, 0.995, 0.997, 0.999, 0.9995, 0.9999]

for graph in graphs:
    exploded = graph.split("_")

    for i in range(5):
        for layer in layers:
            ones = int(int(exploded[5].split(".")[0]) / 2)
            for stepsize in stepsizes:
                for j in range(len(beta1s)):
                    beta1 = beta1s[j]
                    beta2 = beta2s[j]
                    os.system(
                        "python3 with_ent_general_db_copy.py " + str(ones) + " " + graph + " " + str(layer) + " " + str(stepsize) + " " + str(beta1) + " " + str(beta2))
