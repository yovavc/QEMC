import os.path

import pandas as pd

from DbAdapterGraph import DbAdapterGraph
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy import unravel_index
import pickle
from math import log
dbAdapterGraph = DbAdapterGraph()
# rv2 = dbAdapterGraph.get_experiments_graph_numberoflayer_expected_blacks()
# rv3 = dbAdapterGraph.get_experiments_graph_numberoflayer()
# group_by_stepsize_layer = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph()
# group_by_stepsize_layer_256 = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph_256()

# group_by_beta1_layer = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph_beta1()
# global_max_maxcut_by_node_count = dbAdapterGraph.get_maximum_maxcut_group_by_node_number()
# global_avg_maxcut_by_node_count_stepsize = dbAdapterGraph.get_avg_maxcut_group_by_node_number()

# rv = rv3
colors = {
    5: "blue",
    10: "orange",
    30: "brown",
    40: "red",
    60: "black",
    100: "coral",
    200: "green"
}
colors_general = {
    0: "lightcoral",
    1: "indianred",
    2: "brown",
    3: "firebrick",
    4: "maroon",
    5: "darkred",
    6: "green"
}
colors_stepsize = {
    0.02: "slategrey",
    0.04: "orange",
    0.06: "brown",
    0.08: "red",
    0.1: "black",
    0.12: "coral",
    0.14: "green",
    0.16: "pink",
    0.18: "tan",
    0.2: "blue",
    0.3: "darkgrey",
    0.4: "plum",
    0.5: "darkviolet",
    0.6: "navy",
    0.7: "firebrick",
    0.8: "indianred",
}
colors_beta1 = {
    0.9: "blue",
    0.91: "orange",
    0.93: "brown",
    0.95: "red",
    0.97: "black",
    0.99: "coral",
    0.995: "green",
    0.999: "pink",
    0.8: "rosybrown",
    0.81: "darkgrey",
    0.83: "lime",
    0.85: "tan",
    0.87: "slategrey",
    0.89: "navy",
    0.895: "darkviolet",
    0.899: "plum"
}
gw = {8: {"mean": 9.2, "max_cut_gw": 10, "time": 0.1179/10},
      16: {"mean": 45.5, "max_cut_gw": 46, "time": 0.1241/10 },
      32: {"mean": 98.3, "max_cut_gw": 102, "time": 0.1355/10},
      64: {"mean": 199.5, "max_cut_gw": 204, "time": 0.2378/10},
      128: {"mean": 407.4, "max_cut_gw": 415, "time": 0.5834/10},
      256: {"mean": 817.0, "max_cut_gw": 830, "time": 5.3272/10},
      512: {"mean": 1633.8, "max_cut_gw": 1672, "time": 110.3288/10 },
      1024: {"mean": 3285.8, "max_cut_gw": 3307, "time": 1966.1637/10},
      2048: {"mean": 6580.3, "max_cut_gw": 6630, "time": 49234.7973/10},
      3072: {"mean": 9925.0, "max_cut_gw": 9925.0, "time": 148410.232},
      4096: {"mean": "N/A", "max_cut_gw": "N/A"}
      }

font = {'family': 'normal',
        # 'weight': 'bold',
        'size': 14}

plt.rc('font', **font)
q_time= {

}

# data = {}
# collection = dbAdapterGraph.get_experiments_graph_numberoflayer_expected_blacks_by_node_count(256, 50, 0.14)
# plt.clf()
# keys = list(colors_stepsize.keys())
# count = 0
# for entry in collection:
#     current_color = colors_stepsize[keys[int(abs(entry["graphNodeCount"] / 2 - entry["expected_blacks"]))]]
#     rv = dbAdapterGraph.get_max_maxcut_result_set_expected_blacks(entry["ids"])
#     # {"ids": ids, "count": count, "graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer,
#     #                        "stepSize": stepSize, "expected_blacks": expected_blacks}
#     # {"index": iterationIndex, "avg": avg, "std": std}
#     x = []
#     y = []
#     ax = plt.gca()
#     plt.text(0.7, 0.05 + 0.05 * count, entry["expected_blacks"], fontsize=8, horizontalalignment='left',
#              verticalalignment='center',
#              transform=ax.transAxes,
#              color="white",
#              bbox=dict(facecolor=current_color, alpha=1))
#     for set1 in rv:
#         x.append(set1["index"])
#         y.append(set1["avg"])
#     plt.plot(x, y, color=current_color)
#     count += 1
# ax.axhline(y=gw[256]["mean"], color='r', linestyle='--')
# ax.axhline(y=gw[256]["max_cut_gw"], color='g', linestyle='--')
# plt.title(f"nodeCount:256 layers:50 stepSize:0.14 ")
# plt.grid(visible=True)
# plt.ylabel('Maxcut')
# plt.xlabel('Iterations')
# plt.show()
# plt.clf()
# exit()

def get_top_max_index(num, ar):
    rv = []
    for i in range(num):
        rv.append([])
    for i in range(num):
        result = np.where(np.abs(ar) == np.amax(np.abs(ar)))

        # zip the 2 arrays to get the exact coordinates
        listOfCordinates = list(zip(result[0], result[1]))
        rv[i] = listOfCordinates
        for cord in listOfCordinates:
            ar[cord[0], cord[1]] = 0
    print("=============================================")
    for i in range(num):
        print(rv[i])
    print("=============================================")
    return rv

def drawPColor1000(nodes):
    data = {}
    layerCount = 0
    absolute_max = dbAdapterGraph.absolute_max1000(nodes)
    group_by_stepsize_layer_256 = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph_by_nodes_1000(nodes)
    for entry in group_by_stepsize_layer_256:

        rv8 = dbAdapterGraph.get_max_maxcut_result_set_1000(entry["ids"])[0]
        layers = rv8["layers"]
        stepSize = rv8["stepSize"]
        graph = rv8["graph"]
        maxcut = rv8["maxAvgMaxcut"]
        if graph not in data:
            data[graph] = []
        if maxcut is not None:
            data[graph].append((layers, stepSize, maxcut))


    # count(*), stepSize


    for key in data:
        # make these smaller to increase the resolution
        dx, dy = 10, 0.02
        x_boundary = 200
        y_boundary = 0.2
        # generate 2 2d grids for the x & y bounds
        # y, x = np.mgrid[slice(0, y_boundary + dy, dy),
        #                 slice(0, x_boundary + dx, dx)]

        y_values = dbAdapterGraph.get_all_stepsize()
        x_values = dbAdapterGraph.get_all_layers()
        y_values.remove(0.15)
        y_values.remove(0.13)
        x, y = np.mgrid[:len(y_values)+1, :len(x_values)+1]
        # z = np.zeros(y.shape)

        z = np.zeros(y.shape)
        for triple in data[key]:
            shape = z.shape
            z[y_values.index(triple[1]), x_values.index(triple[0])] = triple[2]

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        # (15 / 16)
        z_min, z_max = np.abs(z).max() * 0.9, np.abs(z).max()
        z_temp = z.copy()
        pyramid = 10
        cord_pyramid = get_top_max_index(pyramid, z_temp)

        # maxIndex = unravel_index(z.argmax(), z.shape)
        fig, ax = plt.subplots()
        c = ax.pcolor(x, y, z, cmap='GnBu', vmin=z_min, vmax=z_max)
        ax.set_title(key)
        fig.colorbar(c, ax=ax)
        plt.ylabel('Layers')
        plt.xlabel('Step size')

        minCord = cord_pyramid[0][0]

        for i in range(len(cord_pyramid)):
            cord_set = cord_pyramid[i]
            for cord in cord_set:
                if i == 0:
                    if minCord[1] >= cord[1]:
                        minCord = cord
                # print(cord, z[cord[0], cord[1]])
                plt.text(cord[0], cord[1], f"{round(z[cord[0], cord[1]],2)}", fontsize=8, horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transData,
                         color="white",
                         bbox=dict(facecolor='maroon', alpha=1 - (1/(pyramid+1)) * i))
        plt.text(minCord[0], minCord[1], f"{round(z[minCord[0], minCord[1]],2)}", fontsize=8, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transData,
                 color="white",
                 bbox=dict(facecolor='red', alpha=1))
        global_max = absolute_max[0]["value"]
        plt.text(0.0, -0.1,
                 f"Best set when Step size {y_values[minCord[0]]}, Layers {str(x_values[minCord[1]])}, avgMax:{round(z_max, 2)},max:{global_max}, GWmax:{gw[nodes]['max_cut_gw']}, GWavg:{gw[nodes]['mean']}",
                 fontsize=8, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 color="white",
                 bbox=dict(facecolor='brown', alpha=1))
        # ax = axs[0, 1]
        # c = ax.pcolormesh(x, y, z, cmap='GnBu', vmin=z_min, vmax=z_max)
        # ax.set_title(key)
        # fig.colorbar(c, ax=ax)
        #
        # ax = axs[1, 1]
        # c = ax.pcolorfast(x, y, z, cmap='GnBu', vmin=z_min, vmax=z_max)
        # ax.set_title('pcolorfast')
        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        plt.xticks(range(len(y_values)), y_values)
        plt.yticks(range(len(x_values)), x_values)
        plt.show()
        plt.clf()
        avgMax = z_max
        stepSize = y_values[minCord[0]]
        layers = x_values[minCord[1]]
        return (global_max, avgMax, gw[nodes]['max_cut_gw'], gw[nodes]['mean'], nodes, stepSize, layers)


def drawPColor(nodes):

    data = {}
    layerCount = 0
    absolute_max = dbAdapterGraph.absolute_max(nodes)
    group_by_stepsize_layer_256 = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph_by_nodes(nodes)
    for entry in group_by_stepsize_layer_256:

        rv8 = dbAdapterGraph.get_max_maxcut_result_set(entry["ids"])[0]
        layers = rv8["layers"]
        stepSize = rv8["stepSize"]
        graph = rv8["graph"]
        maxcut = rv8["maxAvgMaxcut"]
        if graph not in data:
            data[graph] = []
        if maxcut is not None:
            data[graph].append((layers, stepSize, maxcut))


    # count(*), stepSize


    for key in data:
        # make these smaller to increase the resolution
        dx, dy = 10, 0.02
        x_boundary = 200
        y_boundary = 0.2
        # generate 2 2d grids for the x & y bounds
        # y, x = np.mgrid[slice(0, y_boundary + dy, dy),
        #                 slice(0, x_boundary + dx, dx)]

        y_values = dbAdapterGraph.get_all_stepsize()
        x_values = dbAdapterGraph.get_all_layers()
        y_values.remove(0.15)
        y_values.remove(0.13)
        x, y = np.mgrid[:len(y_values)+1, :len(x_values)+1]
        # z = np.zeros(y.shape)

        z = np.zeros(y.shape)
        for triple in data[key]:
            shape = z.shape
            z[y_values.index(triple[1]), x_values.index(triple[0])] = triple[2]

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        # (15 / 16)
        z_min, z_max = np.abs(z).max() * 0.9, np.abs(z).max()
        z_temp = z.copy()
        pyramid = 100
        cord_pyramid = get_top_max_index(pyramid, z_temp)

        # maxIndex = unravel_index(z.argmax(), z.shape)
        fig, ax = plt.subplots()
        c = ax.pcolor(x, y, z, cmap='viridis', vmin=z_min, vmax=z_max, edgecolors='w', linewidths=4)

        # ax.set_title(key)
        fig.colorbar(c, ax=ax)
        plt.ylabel('# Layers')
        plt.xlabel('Step size')

        minCord = cord_pyramid[0][0]

        for i in range(len(cord_pyramid)):
            cord_set = cord_pyramid[i]
            for cord in cord_set:
                if i == 0:
                    if minCord[1] >= cord[1]:
                        minCord = cord
                # print(cord, z[cord[0], cord[1]])
                plt.text(cord[0] + 0.5, cord[1] + 0.5, f"{round(z[cord[0], cord[1]],2)}", fontsize=8, horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transData,
                         color="white",
                         bbox=dict(facecolor='maroon', alpha=1 - (1/(pyramid+1)) * i))

        plt.text(minCord[0] +0.1 , minCord[1]+0.5, f"{round(z[minCord[0], minCord[1]],2)}", fontsize=22, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transData,
                 color="black",
                 # bbox=dict(facecolor='red', alpha=1)
                 )
        dd = {"y_values": y_values, "x_values": x_values, "x": x, "y": y, "z": z, "minCord":minCord, "z_min":z_min, "z_max":z_max}

        with open(os.path.basename(__file__) + ".pColor_256_nodes.pickle", 'wb') as handle:
            pickle.dump(dd, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # df = pd.DataFrame(dd)
        # df.to_pickle(os.path.basename(__file__) + ".pColor_256_nodes.pkl")
        global_max = absolute_max[0]["value"]
        # plt.text(0.0, -0.1,
        #          f"Best set when Step size {y_values[minCord[0]]}, Layers {str(x_values[minCord[1]])}, avgMax:{round(z_max, 2)},max:{global_max}, GWmax:{gw[nodes]['max_cut_gw']}, GWavg:{gw[nodes]['mean']}",
        #          fontsize=8, horizontalalignment='left',
        #          verticalalignment='center',
        #          transform=ax.transAxes,
        #          color="white",
        #          bbox=dict(facecolor='brown', alpha=1))
        # ax = axs[0, 1]
        # c = ax.pcolormesh(x, y, z, cmap='GnBu', vmin=z_min, vmax=z_max)
        # ax.set_title(key)
        # fig.colorbar(c, ax=ax)
        #
        # ax = axs[1, 1]
        # c = ax.pcolorfast(x, y, z, cmap='GnBu', vmin=z_min, vmax=z_max)
        # ax.set_title('pcolorfast')
        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        plt.xticks(range(len(y_values)), y_values)
        plt.yticks(range(len(x_values)), x_values)
        plt.show()
        plt.clf()
        avgMax = z_max
        stepSize = y_values[minCord[0]]
        layers = x_values[minCord[1]]
        return (global_max, avgMax, gw[nodes]['max_cut_gw'], gw[nodes]['mean'], nodes, stepSize, layers)



def draw_relative_graph(data_sets):

    fig, axs = plt.subplots(2)

    ax = plt.gca()
    x = []
    relative_y_max = []
    relative_y_avg= []
    for result_set in data_sets:
        global_max = result_set[0]
        avg_max = result_set[1]
        gw_max = result_set[2]
        gw_avg = result_set[3]
        nodes = result_set[4]
        x.append(nodes)
        relative_y_max.append(global_max / gw_max)
        relative_y_avg.append(avg_max / gw_avg)
    # plt.plot(list(range(len(x))), relative_y_max, color='r', linestyle=':', marker="s", label=r'$\frac{Max(QEMCS)}{Max(GW)}$')
    # plt.plot(list(range(len(x))), relative_y_avg, color='b', linestyle=':', marker="*", label=r'$avg(\frac{maxcut_{QEMCS}}{maxcut_{GW}})$')
    axs[0].plot(list(range(len(x))), relative_y_max, color='r', linestyle=':', marker="s", markersize=22, label=r'$\frac{Max(QEMC)}{Max(GW)}$')
    axs[1].plot(list(range(len(x))), relative_y_avg, color='b', linestyle=':', marker="*", markersize=22, label=r'$\frac{Avg(QEMC)}{Avg(GW)}$')
    # plt.title(f"Qmax/GWmax and Qavg/GWavg vs graph vertices count")
    axs[0].set_xticks(range(len(x)), [str(2**4) + " [4]",
                                      str(2**5) + " [5]",
                                      str(2**6) + " [6]",
                                      str(2**7) + " [7]",
                                      str(2**8) + " [8]",
                                      str(2**9) + " [9]",
                                      str(2**10) + " [10]",
                                      str(2**11) + " [11]"])
    axs[1].set_xticks(range(len(x)), [str(2**4) + " [4]",
                                      str(2 ** 5) + " [5]",
                                      str(2 ** 6) + " [6]",
                                      str(2 ** 7) + " [7]",
                                      str(2 ** 8) + " [8]",
                                      str(2 ** 9) + " [9]",
                                      str(2 ** 10) + " [10]",
                                      str(2 ** 11) + " [11]"])
    axs[0].set_ylabel("Relative maximum\n performance")
    axs[1].set_ylabel("Relative average\n performance")
    # axs[0].set_xlabel(r'#Nodes(N)[#qubits($logN$)]')
    axs[1].set_xlabel(r'#Nodes(N) [#qubits($logN$)]')
    axs[0].grid(visible=True)
    axs[1].grid(visible=True)

    leg = axs[0].legend(loc='center')
    leg = axs[1].legend(loc='upper right')
    # plt.text(0.0, 0.1,
    #          f"Qmax/GWmax",
    #          fontsize=8, horizontalalignment='left',
    #          verticalalignment='center',
    #          transform=ax.transAxes,
    #          color="white",
    #          bbox=dict(facecolor='g', alpha=1))
    # plt.text(0.0, 0.3,
    #          f"Qavg/GWavg",
    #          fontsize=8, horizontalalignment='left',
    #          verticalalignment='center',
    #          transform=ax.transAxes,
    #          color="white",
    #          bbox=dict(facecolor='b', alpha=1))
    axs[0].axhline(y=1, color='r', linestyle='--')
    axs[1].axhline(y=1, color='b', linestyle='--')
    plt.show()
    plt.clf()
    return list(range(len(x))), relative_y_max, relative_y_avg

def draw_single_nodes(nodes):
    data = {}
    ax = plt.gca()
    group_by_stepsize_layer_1024 = dbAdapterGraph.get_experiments_group_by_stepsize_layers_graph_1024(nodes)
    for entry in group_by_stepsize_layer_1024:
        rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
        x = []
        y = []
        graph = entry["graph"]
        layers = entry["layers"]
        graphNodeCount = entry["graphNodeCount"]
        stepSize = entry['stepSize']
        if graphNodeCount not in data:
            data[graphNodeCount] = {}
        if layers not in data[graphNodeCount]:
            data[graphNodeCount][layers] = {}
        if stepSize not in data[graphNodeCount][layers]:
            data[graphNodeCount][layers][stepSize] = {}

        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageMaxcut"])

        data[graphNodeCount][layers][stepSize]["x"] = x
        data[graphNodeCount][layers][stepSize]["y"] = y
    for entry in group_by_stepsize_layer_1024:
        rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
        x = []
        y = []
        graph = entry["graph"]
        layers = entry["layers"]
        graphNodeCount = entry["graphNodeCount"]
        stepSize = entry['stepSize']
        if graphNodeCount not in data:
            data[graphNodeCount] = {}
        if layers not in data[graphNodeCount]:
            data[graphNodeCount][layers] = {}
        if stepSize not in data[graphNodeCount][layers]:
            data[graphNodeCount][layers][stepSize] = {}

        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageMaxcut"])

        data[graphNodeCount][layers][stepSize]["x"] = x
        data[graphNodeCount][layers][stepSize]["y"] = y
    for graphNodeCount in data:
        for layers in data[graphNodeCount]:
            ax = plt.gca()
            ax.axhline(y=gw[graphNodeCount]["mean"], color='r', linestyle='--')
            ax.axhline(y=gw[graphNodeCount]["max_cut_gw"], color='g', linestyle='--')
            plt.ylabel('MaxCut')
            plt.xlabel('Iterations')
            plt.title(f"nodes:{graphNodeCount} layers:{layers}")
            for stepSize in data[graphNodeCount][layers]:
                plt.plot(data[graphNodeCount][layers][stepSize]["x"], data[graphNodeCount][layers][stepSize]["y"],
                         color=colors_stepsize[round(stepSize, 2)])
            count = 0

            for item in colors_stepsize.items():
                plt.text(0.8, 0.1 + 0.04 * count, f"Step size {str(item[0])}", fontsize=8, horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transAxes,
                         color="white",
                         bbox=dict(facecolor=item[1], alpha=1))
                count += 1
            plt.grid(visible=True)

            plt.show()
            plt.clf()


def drawMaxAvgAndMax(layers, stepSize, nodes):
    collection = dbAdapterGraph.experiments_by_layer_stepsize2(layers, stepSize, nodes)
    currentGraph = ""
    currentLayer = 0
    print(collection)
    gwIntersectionIteration = 0
    gwAvg = gw[nodes]["mean"]
    for entry in collection:
        rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
        x = []
        y = []
        for record in rv5:
            x.append(record["iteration"])
            if record["averageMaxcut"] is not None and gwAvg >= float(record["averageMaxcut"]):
                gwIntersectionIteration = record["iteration"]

            y.append(record["averageMaxcut"])
        plt.plot(x, y, color="royalblue", label=r'$Average(maxcut_{QEMCS})$')
    for entry in collection:
        # rv5 = dbAdapterGraph.get_maxcut_result_set("16230")
        rv5 = dbAdapterGraph.get_maxcut_result_set("14747")
        x = []
        y = []
        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageMaxcut"])
        plt.plot(x, y, color='blue', label=r'$Maximum(maxcut_{QEMCS})$')
    ax = plt.gca()
    ax.axhline(y=gw[nodes]["mean"], color='pink', linestyle='--', )
    ax.axhline(y=gw[nodes]["max_cut_gw"], color='fuchsia', linestyle='--')
    ax.axvline(x=gwIntersectionIteration, color='blue', linestyle='--')
    leg = plt.legend(loc='lower right')
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 max_Maxcut: 833")
    plt.ylabel('MaxCut')
    plt.xlabel('# Iterations')
    plt.grid(visible=True)
    plt.show()
    plt.clf()
    return gwIntersectionIteration

def drawMaxAvgAndMax1000(layers, stepSize, nodes):
    collection = dbAdapterGraph.experiments_by_layer_stepsize2(layers, stepSize, nodes)
    currentGraph = ""
    currentLayer = 0
    print(collection)
    gwIntersectionIteration = 0
    gwAvg = gw[nodes]["mean"]
    for entry in collection:
        rv5 = dbAdapterGraph.get_maxcut_result_set1000(entry["ids"])
        x = []
        y = []
        for record in rv5:
            x.append(record["iteration"])
            if gwAvg >= record["averageMaxcut"]:
                gwIntersectionIteration = record["iteration"]

            y.append(record["averageMaxcut"])
        plt.plot(x, y, color="royalblue", label=r'$Average(maxcut_{QEMCS})$')
    # for entry in collection:
    #     # rv5 = dbAdapterGraph.get_maxcut_result_set("16230")
    #     rv5 = dbAdapterGraph.get_maxcut_result_set("14747")
    #     x = []
    #     y = []
    #     for record in rv5:
    #         x.append(record["iteration"])
    #         y.append(record["averageMaxcut"])
    #     plt.plot(x, y, color='blue', label=r'$Maximum(maxcut_{QEMCS})$')
    ax = plt.gca()
    ax.axhline(y=gw[nodes]["mean"], color='gray', linestyle='--', )
    ax.axhline(y=gw[nodes]["max_cut_gw"], color='black', linestyle='--')
    ax.axvline(x=gwIntersectionIteration, color='blue', linestyle='--')
    leg = plt.legend(loc='lower left')
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 max_Maxcut: 833")
    plt.ylabel('MaxCut')
    plt.xlabel('# Iterations')
    plt.grid(visible=True)
    plt.show()
    # plt.clf()
    return gwIntersectionIteration


def drawHWGraph(ids, color='blue', label=None, limit=None, use_title=False):
    # drawMaxAvgAndMax(80, 0.08, 256)
    # rv5 = dbAdapterGraph.get_maxcut_instance_result_set("129159,129158,129153,129152,129151,129150,129149")
    rv5 = dbAdapterGraph.get_maxcut_instance_result_set(ids)
    x = []
    y = []
    name = None
    for record in rv5:
        x.append(record["iteration"])
        y.append(record["averageMaxcut"])
        name = record["graph"]

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    if label is None:
        plt.plot(x, y, color='blue', label=r'$Maximum(maxcut_{QEMCS})$', linewidth=2)
    else:
        plt.plot(x, y, color=color, label=label,  linewidth=2)
    leg = plt.legend(loc='lower right', prop={'size': 10})
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 max_Maxcut: 833")
    plt.ylabel('Cut')
    plt.xlabel('# Iterations')
    plt.grid(visible=True)
    if use_title:
        plt.title(name)
    return x, y
    # draw_single_nodes(256)
    # draw_single_nodes(1024)
    # drawPColor(8), drawPColor(16),
    # Step 2

def drawHWGraphLoss(ids, color='blue', label=None, limit=None, use_title=False):
    # drawMaxAvgAndMax(80, 0.08, 256)
    # rv5 = dbAdapterGraph.get_maxcut_instance_result_set("129159,129158,129153,129152,129151,129150,129149")
    rv5 = dbAdapterGraph.get_loss_result_set(ids)
    x = []
    y = []
    name = None
    for record in rv5:
        x.append(record["iteration"])
        y.append(record["averageLoss"])
        name = record["graph"]

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    if label is None:
        plt.plot(x, y, color='blue', label=r'$Average(maxcut_{QEMCS})$', linewidth=2)
    else:
        plt.plot(x, y, color=color, label=label,  linewidth=2)
    leg = plt.legend(loc='lower right', prop={'size': 10})
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 max_Maxcut: 833")
    plt.ylabel('Cut')
    plt.xlabel('# Iterations')
    plt.grid(visible=True)
    if use_title:
        plt.title(name)
    return x, y
    # draw_single_nodes(256)
    # draw_single_nodes(1024)
    # drawPColor(8), drawPColor(16),
    # Step 2
if __name__ == '__main__':

    # for entry in ['129804','129803','129802','129799','129798','129797','129796','129795','129794','129793',
    #               '129792','129790','129789','129787','129786','129785']:
    #     drawHWGraph(entry)
    # drawHWGraph("129770")
    # 8 nodes

    # '129785', '46'
    # '129786', '251'
    # '129787', '251'
    # '129854', '42'
    # '129855', '56'
    # '129856', '42'

    drawHWGraph("129785, 129786, 129787, 129854, 129855", 'black', 'HW', 50)
    drawHWGraph("129840,129841,129842,129843,129844", 'blue', "Simulator", 50)
    plt.show()


    # 16 nodes
    drawHWGraph('129857, 129860, 129858, 129859','red', 'HW')
    drawHWGraph('129865,129864,129863,129862,129861', 'blue', "Simulator")
    ax = plt.gca()
    ax.axhline(y=21, color='gray', linestyle='--', label="Optimal cut")
    leg = plt.legend(loc='lower right')
    plt.show()


    # exit()
    with open('results_for_scale_graph', 'rb') as results_for_scale_graph_file:
        # Step 3
        results_for_scale_graph = pickle.load(results_for_scale_graph_file)
    x, y_max, y_avg = draw_relative_graph(results_for_scale_graph)
    dd = {"x": x, "y_max": y_max, "y_avg": y_avg}
    df = pd.DataFrame(dd)
    df.to_csv(os.path.basename(__file__) + ".relative_plot.csv")
    intersections = []
    some_times = {}
    some_times2 = {}
    for result_set in results_for_scale_graph:
        global_max = result_set[0]
        avg_max = result_set[1]
        gw_max = result_set[2]
        gw_avg = result_set[3]
        nodes = result_set[4]
        stepSize = result_set[5]
        layers = result_set[6]

        if nodes < 512:
            intersections.append((nodes, drawMaxAvgAndMax(layers, stepSize, nodes), layers))
        else:
            intersections.append((nodes, drawMaxAvgAndMax1000(layers, stepSize, nodes), layers))
        some_times[nodes] = float(dbAdapterGraph.get_time(nodes, layers)[0]) * intersections[-1][1]
        some_times2[nodes] = float(dbAdapterGraph.get_time(nodes, layers)[0])
    x = []
    y2 =[]
    y = []
    for intersection in intersections:
        x.append(intersection[0])
        y.append(intersection[1])
        y2.append(intersection[2])
    plt.plot(range(len(x)), y, marker="s", markersize=22, label=r'GW average intersection')
    dd = {"x":range(len(x)), "y":y}
    df = pd.DataFrame(dd)
    df.to_csv(os.path.basename(__file__) + ".GW_average_intersection_iterations_function_of_nodes.csv")

    # plt.xticks(range(len(x)), x)
    plt.xticks(range(len(x)), [str(2 ** 4) + "(4)",
                                      str(2 ** 5) + "\n(5)",
                                      str(2 ** 6) + "\n(6)",
                                      str(2 ** 7) + "\n(7)",
                                      str(2 ** 8) + "\n(8)",
                                      str(2 ** 9) + "\n(9)",
                                      str(2 ** 10) + "\n(10)",
                                      str(2 ** 11) + "\n(11)"])
    plt.xlabel(r'#nodes[$2^n$] (#qubits[n])')
    plt.ylabel(r'#Iterations')
    plt.grid(visible=True)
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()

    plt.plot(range(len(x)), y2, marker="s", markersize=22, label=r'GW average intersection')
    dd = {"x": range(len(x)), "y": y2}
    df = pd.DataFrame(dd)
    df.to_csv(os.path.basename(__file__) + ".GW_average_intersection_layer_function_of_nodes.csv")
    plt.xticks(range(len(x)), [str(2 ** 4) + "(4)",
                                      str(2 ** 5) + "\n(5)",
                                      str(2 ** 6) + "\n(6)",
                                      str(2 ** 7) + "\n(7)",
                                      str(2 ** 8) + "\n(8)",
                                      str(2 ** 9) + "\n(9)",
                                      str(2 ** 10) + "\n(10)",
                                      str(2 ** 11) + "\n(11)"])
    plt.xlabel(r'#nodes[$2^n$] (#qubits[n])')
    plt.ylabel(r'#Layers')
    plt.grid(visible=True)
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()


    drawPColor(2048)
    drawPColor1000(2048)
    drawPColor(256)
    exit()
    # import math
    # x= []
    # y= []
    # y2= []
    # y3 = []
    # y4 =[]
    # for key in gw.keys():
    #     if key == 2048 or key == 4096 or key == 8:
    #         continue
    #     x.append(key)
    #     y.append(gw[key]["time"])
    #     y2.append(some_times[key])
    #     y3.append(some_times2[key])
    # mymodel = np.poly1d(np.polyfit(x, y, 3))
    # mymodel2 = np.poly1d(np.polyfit(x, y2, 2))
    # x2 =[]
    # myline = np.linspace(8, 4096, 100)
    # some_times2[2048] = float(dbAdapterGraph.get_time(2048, 120)[0])
    # some_times2[4096] = float(dbAdapterGraph.get_time(4096, 150)[0])
    # # x2 = [16, 32, 54, 128, 256,512, 1024,2048, 4096]
    # for key in some_times2:
    #    y4.append(some_times2[key])
    #    x2.append(key)
    # plt.scatter(np.log2(x), y, s=300)
    # # plt.scatter(x, y2, s=300)
    # plt.scatter(np.log2(x), y3, s=300)
    # plt.scatter(np.log2(x2),y4, s=300)
    # # plt.plot(myline, mymodel(myline))
    # # plt.plot(myline, mymodel2(myline))
    # plt.show()
    # plt.clf()
    #
    # x =[]
    # y= []
    #
    #
    #
    # plt.scatter(x, y, s=300)
    # # plt.scatter(x, y2, s=300)
    # plt.scatter(x, y3, s=300)
    # exit()
    drawPColor(16)
    drawPColor(256)
    results_for_scale_graph = [
        drawPColor(16),
        drawPColor(32),
        drawPColor(64),
        drawPColor(128),
        drawPColor(256),
        drawPColor1000(512),
        drawPColor1000(1024),
        drawPColor1000(2048)
        ]
    # # Step 2
    # with open('results_for_scale_graph', 'wb') as results_for_scale_graph_file:
    #     # Step 3
    #     pickle.dump(results_for_scale_graph, results_for_scale_graph_file)
    # draw_relative_graph(results_for_scale_graph)

    # collection = dbAdapterGraph.experiments_by_layer_stepsize2(50, 0.14, 256)
    # currentGraph = ""
    # currentLayer = 0
    # for entry in collection:
    #     rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
    #     x = []
    #     y = []
    #     for record in rv5:
    #         x.append(record["iteration"])
    #         y.append(record["averageMaxcut"])
    #     plt.plot(x, y, color="r")
    # for entry in collection:
    #     rv5 = dbAdapterGraph.get_maxcut_result_set("16230")
    #     x = []
    #     y = []
    #     for record in rv5:
    #         x.append(record["iteration"])
    #         y.append(record["averageMaxcut"])
    #     plt.plot(x, y, color='g')
    # ax = plt.gca()
    # ax.axhline(y=gw[256]["mean"], color='r', linestyle='--')
    # ax.axhline(y=gw[256]["max_cut_gw"], color='g', linestyle='--')
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 max_Maxcut: 833")
    # plt.ylabel('MaxCut')
    # plt.xlabel('Iterations')
    # plt.grid(visible=True)
    # plt.show()
    # plt.clf()
    # exit()
    # exit()

    collection = dbAdapterGraph.experiments_by_layer_stepsize2(80, 0.08, 256)
    currentGraph = ""
    currentLayer = 0
    for entry in collection:
        rv5 = dbAdapterGraph.get_loss_result_set(entry["ids"])
        x = []
        y = []
        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageLoss"])
        plt.plot(x, y, color="royalblue", label=r'$Avarage \, run \, (loss_{QEMCS})$')
    for entry in collection:
        rv5 = dbAdapterGraph.get_loss_result_set("14747")
        x = []
        y = []
        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageLoss"])
        plt.plot(x, y, color='blue', label=r'$Maximum \, run \, (loss_{QEMCS})$')
    ax = plt.gca()
    # ax.axhline(y=gw[256]["mean"], color='r', linestyle='--')
    # ax.axhline(y=gw[256]["max_cut_gw"], color='g', linestyle='--')
    # plt.title(f"nodeCount:256 layers:50 stepSize:0.14 Loss")
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    leg = plt.legend(loc='upper left')
    plt.grid(visible=True)
    # plt.text(0.0, -0.05, f"Avg", fontsize=8, horizontalalignment='left',
    #          verticalalignment='center',
    #          transform=ax.transAxes,
    #          color="white",
    #          bbox=dict(facecolor='r', alpha=1))
    # plt.text(0.0, -0.1, f"Max", fontsize=8, horizontalalignment='left',
    #          verticalalignment='center',
    #          transform=ax.transAxes,
    #          color="white",
    #          bbox=dict(facecolor='g', alpha=1))
    plt.show()
    plt.clf()
    #
    # currentGraph = ""
    # currentLayer = 0
    # for entry in group_by_beta1_layer:
    #     rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
    #     x = []
    #     y = []
    #     for record in rv5:
    #         x.append(record["iteration"])
    #         y.append(record["averageMaxcut"])
    #     plt.plot(x, y, color=colors_beta1[entry['beta1']])
    # count = 0
    # for item in colors_beta1.items():
    #     ax = plt.gca()
    #     plt.title(f"{entry['graph']} layers:{entry['layers']}")
    #     plt.ylabel('MaxCut')
    #     plt.xlabel('Iterations')
    #     plt.text(0.8, 0.1 + 0.035 * count, f"beta1 {str(item[0])}", fontsize=8, horizontalalignment='left',
    #                  verticalalignment='center',
    #                  transform=ax.transAxes,
    #                  color="white",
    #                  bbox=dict(facecolor=item[1], alpha=1))
    #     count += 1
    # plt.grid(visible=True)
    # plt.show()
    # plt.clf()
    # exit()
    #


    draw_single_nodes

    x = []
    y = []
    for entry in global_max_maxcut_by_node_count:
        graphNodeCount = entry["graphNodeCount"]
        globalMaxMaxcut = entry["globalMaxMaxcut"]
        if graphNodeCount in gw:
            x.append(graphNodeCount)
            y.append(globalMaxMaxcut / gw[graphNodeCount]["max_cut_gw"])
    plt.title(f"Qmax/GWmax vs graph vertices count")
    plt.ylabel('Qmax/GWmax')
    plt.xlabel('vertices')
    ax.axhline(y=1, color='r', linestyle='--')
    plt.grid(visible=True)
    plt.plot(x, y, marker='o')


    plt.show()
    plt.clf()

    ax = plt.gca()
    x = []
    y = []
    for entry in global_avg_maxcut_by_node_count_stepsize:
        graphNodeCount = entry["graphNodeCount"]
        globalMaxMaxcut = entry["globalMaxMaxcut"]
        if graphNodeCount in gw:
            x.append(graphNodeCount)
            y.append(globalMaxMaxcut / gw[graphNodeCount]["mean"])
    plt.title(f"Qmean/GWmean vs graph vertices count")
    plt.ylabel('Qmean/GWmean')
    plt.xlabel('vertices')
    ax.axhline(y=1, color='r', linestyle='--')
    plt.grid(visible=True)
    plt.plot(x, y, marker='o')

    plt.show()
    plt.clf()
    exit()

    print(group_by_stepsize_layer)
    currentGraph = ""
    currentLayer = 0
    for entry in group_by_stepsize_layer:
        rv5 = dbAdapterGraph.get_maxcut_result_set(entry["ids"])
        x = []
        y = []
        if currentGraph != entry["graph"] or currentLayer != entry["layers"]:
            if currentLayer != 0:
                count = 0
                ax = plt.gca()
                graphNodeCount = entry["graphNodeCount"]
                if graphNodeCount in gw:
                    ax.axhline(y=gw[graphNodeCount]["mean"], color='r', linestyle='--')
                    ax.axhline(y=gw[graphNodeCount]["max_cut_gw"], color='g', linestyle='--')
                for item in colors_stepsize.items():
                    plt.title(f"{entry['graph']} layers:{entry['layers']}")
                    plt.ylabel('MaxCut')
                    plt.xlabel('Iterations')
                    plt.text(0.8, 0.1 + 0.07 * count, f"Step size {str(item[0])}", fontsize=8, horizontalalignment='left',
                             verticalalignment='center',
                             transform=ax.transAxes,
                             color="white",
                             bbox=dict(facecolor=item[1], alpha=1))
                    count += 1
                plt.grid(visible=True)
                plt.show()
                plt.clf()
            currentGraph = entry["graph"]
            currentLayer = entry["layers"]

        for record in rv5:
            x.append(record["iteration"])
            y.append(record["averageMaxcut"])
        plt.plot(x, y, color=colors_stepsize[entry['stepSize']])
    exit()

    x1 = []
    y1 = []
    for entry in rv2:
        ids = entry["eid"]
        ids = ids.split(",")
        for e_id in ids:
            x1.append([])
            y1.append([])
            rv = dbAdapterGraph.get_single_experiment_maxcut(int(e_id))
            print(rv)
            for record in rv:
                x1[-1].append(record["iterationIndex"])
                y1[-1].append(record["maxcut"])
            plt.plot(x1[-1], y1[-1])
        plt.title(f"{entry['graph']} layers:{entry['numberOfLayers']} eBlacks:{entry['expectedBlacks']}")
        plt.ylabel('MaxCut')
        plt.xlabel('Iterations')
        plt.grid(visible=True)
        plt.savefig(f"{entry['graph']}_layers_{entry['numberOfLayers']}_eBlacks_{entry['expectedBlacks']}.png")
        plt.show()
        plt.clf()
    exit()
    currentGraph = ""
    figure, axis = plt.subplots(1, 3)
    for entry in rv2:

        if entry["graph"] != currentGraph:
            currentGraph = entry["graph"]
            count = 0
            for item in colors.items():
                axis[0].text(-0.2, 0.6 + 0.05 * count, f"Layer {str(item[0])}", fontsize=8, horizontalalignment='left',
                             verticalalignment='center',
                             transform=axis[0].transAxes,
                             color="white",
                             bbox=dict(facecolor=item[1], alpha=1))
                count += 1
            plt.show()
            width = 16
            height = 16 * 9 / width
            plt.rcParams["figure.figsize"] = [width, height]
            plt.rcParams["figure.autolayout"] = True
            plt.savefig(f"results/grouped/{entry['graph']}_grouped_layers.png")
            figure, axis = plt.subplots(1, 3)

        # print(entry["eid"])
        rv1 = dbAdapterGraph.get_maxcut_result_set(entry["eid"])
        x = []
        min_maxcut = []
        max_maxcut = []
        avg_maxcut = []
        for entry2 in rv1:
            x.append(entry2["iteration"])
            min_maxcut.append(entry2["minMaxcut"])
            max_maxcut.append(entry2["maxMaxcut"])
            avg_maxcut.append(entry2["averageMaxcut"])

        figure.suptitle(entry['graph'])
        axis[0].set_title(f"Min")
        axis[1].set_title(f"Max")
        axis[2].set_title(f"Avg")

        axis[0].set_ylabel('MaxCut')
        axis[0].set_xlabel('Iterations')
        axis[0].grid(visible=True)

        axis[1].set_ylabel('MaxCut')
        axis[1].set_xlabel('Iterations')
        axis[1].grid(visible=True)

        axis[2].set_ylabel('MaxCut')
        axis[2].set_xlabel('Iterations')
        axis[2].grid(visible=True)

        axis[0].plot(x, min_maxcut, color=colors[entry['numberOfLayers']], label=entry['numberOfLayers'])
        axis[1].plot(x, max_maxcut, color=colors[entry['numberOfLayers']], label=entry['numberOfLayers'])
        axis[2].plot(x, avg_maxcut, color=colors[entry['numberOfLayers']], label=entry['numberOfLayers'])
        # figure.tight_layout()

        # plt_file = directory + "/" + "avg.plt" + ".png"
        # plt.axhline(y=mean_gw, color='r', linestyle='--')
        # plt.axhline(y=avg_gw, color='g', linestyle='--')
        # # plt.savefig(plt_file)
        # plt.clf()

    # plt.show()
    plt.clf()
    for entry in rv3:
        print(entry["eid"])
        rv1 = dbAdapterGraph.get_maxcut_result_set(entry["eid"])
        x = []
        min_maxcut = []
        max_maxcut = []
        avg_maxcut = []
        for entry2 in rv1:
            x.append(entry2["iteration"])
            min_maxcut.append(entry2["minMaxcut"])
            max_maxcut.append(entry2["maxMaxcut"])
            avg_maxcut.append(entry2["averageMaxcut"])

        plt.ylabel('MaxCut')
        plt.xlabel('Iterations')
        plt.title(f"{entry['graph']} layers:{entry['numberOfLayers']}")
        plt.grid(visible=True)

        plt.plot(x, min_maxcut, color="green")
        plt.plot(x, max_maxcut, color="red")
        plt.plot(x, avg_maxcut, color="blue")
        ax = plt.gca()
        plt.text(0.2, 0.1 + 0.1 * 2, f"max", fontsize=8, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 color="white",
                 bbox=dict(facecolor="red", alpha=1))

        plt.text(0.2, 0.1 + 0.1 * 0, f"Min", fontsize=8, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 color="white",
                 bbox=dict(facecolor="green", alpha=1))

        plt.text(0.2, 0.1 + 0.1 * 1, f"avg", fontsize=8, horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 color="white",
                 bbox=dict(facecolor="blue", alpha=1))

        # plt.show()
        plt.savefig(f"results/{entry['graph']}_layer_{entry['numberOfLayers']}.png")
        # plt_file = directory + "/" + "avg.plt" + ".png"
        # plt.axhline(y=mean_gw, color='r', linestyle='--')
        # plt.axhline(y=avg_gw, color='g', linestyle='--')
        # # plt.savefig(plt_file)
        plt.clf()
