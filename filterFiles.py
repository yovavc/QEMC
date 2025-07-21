import os
import glob
import shutil
directory = "with_ent_regular_graph_result"
layers = [2, 5, 10, 30]
graph_sizes = [4, 5, 6, 7, 8, 9, 10]
destination = r"C:\Users\yovav\PycharmProjects\muxcut\groupBy\same_number_of_size_and_layers"
for layer in layers:
    for size in graph_sizes:
        node_count = 2**size
        blacks = node_count/2
        for i in range(0,6):
            black = int(blacks - i)
        #files = glob.glob(directory + "/vertices_regular_9_size_1024.g6_layers_10*")
            files = glob.glob(directory + "/vertices_regular_9_size_" + str(node_count) + ".g6_layers_" + str(layer) + "_ones_" + str(black) +"*")
            dst = destination+"/" + str(node_count) + "_" + str(layer) + "/ones_" + str(black) + "/"
            #os.makedirs(dst, exist_ok=True)
            for file in files:
                #shutil.copy(file, dst)
                print(dst)