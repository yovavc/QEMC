import os
graph = "vertices_16_index_1.g6"

max_cut = 55
for i in range(0,10):
    os.system("python" + " no_ent_general.py " + str(16) + " 8 " + graph + " > no_ent_out_general" + "16_1" + "_" + str(i) + ".txt")