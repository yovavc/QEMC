import os
from chardet import detect
import re
import matplotlib.pyplot as plt


def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


directory = "groupBy\\same_number_of_size_and_layers\\16_2\\ones_5"


def plot_dir(directory, mean_gw, avg_gw):
    files = os.listdir(directory)
    x_all = []
    y_all = []
    files.sort()
    graph_name = None
    record_length = 201
    avg_vector = [0] * record_length
    for file in files:

        graph_name = file.split(".")[0]
        if file == 'avg.plt.png':
            continue
        # reading from file
        file1 = open(directory + "/" + file, 'r', encoding='utf-8', errors='ignore')
        count = 0
        Lines = file1.readlines()
        #print(file)
        file1.close()
        step = None
        cut = None
        max_cut_found_at = None
        latest_max_cut = 0
        x = []
        y = []

        for line in Lines:

            strippedLine = line.strip()
            if re.search("max_cut=", strippedLine):
                cut = float(re.sub(' +', ' ', strippedLine).split(" ")[1])
            if re.search("after step", strippedLine):
                step = int(re.sub(' +', ' ', strippedLine.split(":")[0]).split(" ")[3])
            if re.search("max_cut found at =", strippedLine):
                max_cut_found_at = int(re.sub(' +', ' ', strippedLine).split(" ")[4])
            if cut is not None and step is not None and max_cut_found_at is not None:
                x.append(step)
                y.append(cut)
                latest_max_cut = max_cut_found_at
                # print(cut, step, max_cut_found_at)
                cut = None
                step = None
                max_cut_found_at = None

        for line in Lines:

            strippedLine = line.strip()
            if re.search("max_cut=", strippedLine):
                cut = float(re.sub(' +', ' ', strippedLine).split(" ")[1])
            if re.search("after step", strippedLine):
                step = int(re.sub(' +', ' ', strippedLine.split(":")[0]).split(" ")[3])
            if re.search("max_cut found at =", strippedLine):
                max_cut_found_at = int(re.sub(' +', ' ', strippedLine).split(" ")[4])
            if cut is not None and step is not None and max_cut_found_at is not None:
                if step == latest_max_cut:
                    break
            count += 1
        print(Lines[count-2])
        # x_all.append(x[0:int(latest_max_cut * 1.1)])
        # y_all.append(y[0:int(latest_max_cut * 1.1)])
        #print(len(y))
        x_all.append(x)
        y_all.append(y)
        plt.ylabel('MaxCut')
        plt.xlabel('Iterations')
        plt.title(file + "\n" + graph_name)
        plt.grid(visible=True)

        record_count = 0

        for j in range(record_length):
            for t in range(len(y_all)):
                avg_vector[j] = avg_vector[j] + y_all[t][j]
            avg_vector[j] = avg_vector[j] / len(y_all)

        plt.plot(x_all[-1], avg_vector)

        # plt.show()
        plt_file = directory + "/" + "avg.plt" + ".png"
        plt.axhline(y=mean_gw, color='r', linestyle='--')
        plt.axhline(y=avg_gw, color='g', linestyle='--')
        #plt.savefig(plt_file)
        plt.clf()


layers = [2, 5, 10, 30]
graph_sizes = [4, 5, 6, 7, 8, 9, 10]
destination = r"C:\Users\yovav\PycharmProjects\muxcut\groupBy\same_number_of_size_and_layers"

gw = {16: {"mean": 45.5, "max_cut_gw": 46},
      32: {"mean": 98.3, "max_cut_gw": 102},
      64: {"mean": 199.5, "max_cut_gw": 204},
      128: {"mean": 407.4, "max_cut_gw": 415},
      256: {"mean": 817.0, "max_cut_gw": 830},
      512: {"mean": 1633.8, "max_cut_gw": 1672},
      1024: {"mean": 3285.8, "max_cut_gw": 3307}
      }

for layer in layers:
    for size in graph_sizes:
        node_count = 2 ** size
        blacks = node_count / 2
        for i in range(0, 6):
            black = int(blacks - i)
            dst = destination + "/" + str(node_count) + "_" + str(layer) + "/ones_" + str(black) + "/"
            print(dst)
            plot_dir(dst, gw[node_count]["mean"], gw[node_count]["max_cut_gw"])
