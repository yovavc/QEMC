import os

import math
from chardet import detect
import re_d
import matplotlib.pyplot as plt


def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


directory = "groupBy\\same_number_of_size_and_layers\\16_2\\ones_5"


def plot_dir(directory, black, delta_black, mean_gw, avg_gw, ax ):
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
        # writing to file
        file1 = open(directory + "/" + file, 'r', encoding='utf-8', errors='ignore')
        count = 0
        Lines = file1.readlines()
        print(file)
        file1.close()
        step = None
        cut = None
        max_cut_found_at = None
        latest_max_cut = 0
        x = []
        y = []

        for line in Lines:
            strippedLine = line.strip()
            if re_d.search("max_cut=", strippedLine):
                cut = float(re_d.sub(' +', ' ', strippedLine).split(" ")[1])
            if re_d.search("after step", strippedLine):
                step = int(re_d.sub(' +', ' ', strippedLine.split(":")[0]).split(" ")[3])
            if re_d.search("max_cut found at =", strippedLine):
                max_cut_found_at = int(re_d.sub(' +', ' ', strippedLine).split(" ")[4])
            if cut is not None and step is not None and max_cut_found_at is not None:
                x.append(step)
                y.append(cut)
                latest_max_cut = max_cut_found_at
                # print(cut, step, max_cut_found_at)
                cut = None
                step = None
                max_cut_found_at = None

        # x_all.append(x[0:int(latest_max_cut * 1.1)])
        # y_all.append(y[0:int(latest_max_cut * 1.1)])
        print(len(y))
        x_all.append(x)
        y_all.append(y)

        plt.grid(visible=True)

        record_count = 0


    for j in range(record_length):
        for t in range(len(y_all)):
            avg_vector[j] = avg_vector[j] + y_all[t][j]
        avg_vector[j] = avg_vector[j] / len(y_all)
    colors ={
        0: "blue",
        1: "orange",
        2: "brown",
        3: "sienna",
        4: "black",
        5: "maroon"
    }
    ax.plot(x_all[-1], avg_vector, color=colors[delta_black])
    # plt.show()

    # plt.axhline(y=mean_gw, color='r', linestyle='--')
    # plt.axhline(y=avg_gw, color='g', linestyle='--')

    ax.text(0.8, 0.05 * delta_black + 0.1, str(avg_vector[-1]) + "_" + str(black),transform=ax.transAxes, color=colors[delta_black])

    return x_all, y_all



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
        x_all = {}
        y_all = {}
        mean_gw = gw[node_count]["mean"]
        avg_gw = gw[node_count]["max_cut_gw"]
        fig, ax = plt.subplots()
        for i in range(0, 6):
            black = int(blacks - i)
            dst = destination + "/" + str(node_count) + "_" + str(layer) + "/ones_" + str(black) + "/"
            # print(dst)
            x, y = plot_dir(dst, black, i, mean_gw, avg_gw, ax )
            x_all[black] = x
            y_all[black] = y
            # for k in range(len(x)):
            #     x_all.append(x[k])
            #     y_all.append(y[k])
        max_record_index = 0
        max_maxcut_value = 0
        max_series_key = 0
        for item in y_all.items():

            for t in range(len(item[1])):
                if max_maxcut_value < item[1][t][-1]:
                    max_maxcut_value = item[1][t][-1]
                    max_record_index = t
                    max_series_key = item[0]

        ax.axhline(y=mean_gw, color='r', linestyle='--')
        ax.axhline(y=avg_gw, color='g', linestyle='--')
        ax.plot(x_all[max_series_key][-1], y_all[max_series_key][max_record_index], color="purple", linestyle='solid')
        ax.text(x_all[max_series_key][-1][int((170 / 5) * 5)], y_all[max_series_key][max_record_index][-1], str(y_all[max_series_key][max_record_index][-1]) + "_" + str(max_series_key), color="purple")

        plt.ylabel('MaxCut')
        plt.xlabel('Iterations')
        plt.title("Regular graph 9 with " + str(node_count) + " nodes and QC with " + str(layer) + " layers")
        ax.text(0.5, 0.05, "GW mean:" + str(mean_gw), transform=ax.transAxes, color="red")
        ax.text(0.5, 0.1, "GW max:" + str(avg_gw), transform=ax.transAxes, color="green")
        save_fig = destination + "/" + str(node_count) + "_" + str(layer)
        plt_file = save_fig + "/" + "avg.plt" + ".png"
        plt.savefig(plt_file)
        plt.clf()
