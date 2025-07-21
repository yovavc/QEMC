import argparse
import os.path
import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from DrawGraphFromDb import drawHWGraph, drawHWGraphLoss

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
args = parser.parse_args()


def get_max_cut(graph6):
    graph = nx.read_graph6(graph6)
    edges = list(graph.edges)
    nodes = graph.nodes
    global_max_cut = 0
    global_max_cut_combination = 0
    count = 0
    y = []
    while True:
        i = random.randint(0, 2 ** (len(nodes)))
        if bin(i).count("1") != (len(nodes)/2):
            continue
    # for i in range(0, 2 ** (len(nodes))):

        max_cut_for_i = 0
        for edge in edges:
            mask = (1 << edge[1]) + (1 << edge[0])
            active_bits = i & mask
            is_arc_between_two_groups = (active_bits & (active_bits-1) == 0) and active_bits != 0
            # print(f'{mask:032b}' + " " + f'{i:032b}' + " " + f'{active_bits:032b}' + " " + str(is_arc_between_two_groups))
            if is_arc_between_two_groups:
                max_cut_for_i += 1
        print("current max cut: " + str(max_cut_for_i) + " for combination " + str(i) + " " + bin(i) + " " + str(count))

        if global_max_cut < max_cut_for_i:
            global_max_cut = max_cut_for_i
            global_max_cut_combination = i
        y.append(global_max_cut)
        count += 1
        if max_cut_for_i == 10:
            break

    print("global max cut: " + str(global_max_cut) + " for combination " + str(global_max_cut_combination)
          + " " + bin(global_max_cut_combination))
    print(edges)
    return range(0, count), y

X = []
Y = []
X_padded = []
Y_padded = []
Y_avg = []
max_x = 0
max_y = 0
runs = 5
for i in range(0,runs):
    x, y = get_max_cut(args.fileName)
    X.append(x)
    Y.append(y)
    print(Y)
    if max_y < len(y):
        max_y = len(y)
    if max_x < len(x):
        max_x = len(x)
    print(len(x), len(y))

limit = 50
for y in Y:
    y += [21] * (limit - len(y))
    X_padded.append(range(0, limit))
    Y_padded.append(y[:limit])


print(len(X), len(Y), max_x, max_y)
Y_avg += [0] * (limit)
for y in Y_padded:
    for i in range(0, limit):
        Y_avg[i] += y[i]

for i in range(0, limit):
    Y_avg[i] = Y_avg[i] / runs
# for x,y in zip(X_padded,Y_padded):
#     plt.plot(x, y, color='red', label='random search')

ax = plt.gca()
ax.axhline(y=10, color='green',linestyle='dashed',linewidth=2 , label="Optimal cut")



# x_rh, y_rh  = drawHWGraph("129792,129793,129794,129795,129796", 'red', 'Avg. QEMC: real hardware (IBMQ)', 50) # For 4 node graph

# x_rh, y_rh  = drawHWGraph("129857,129859,129860,129871,129872,129874", 'red', 'Avg. QEMC: real hardware (IBMQ)', 50)  # For 16 node graph
x_ns, y_ns = drawHWGraph("129840,129841,129842,129843,129844", 'blue', "Avg. QEMC: noiseless simulation", 50)
x_rh, y_rh = drawHWGraph("129785, 129786, 129787, 129854, 129855", 'red', 'Avg. QEMC: real hardware (IBMQ)', 50) # For 8 nodes graph
x_nsl, y_nsl = drawHWGraphLoss("129840,129841,129842,129843,129844", 'blue', "Avg. QEMC: noiseless simulation", 50)
x_rhl, y_rhl = drawHWGraphLoss("129785, 129786, 129787, 129854, 129855", 'red', 'Avg. QEMC: real hardware (IBMQ)', 50)

dd = {"x_random":  X_padded[0], "y_random": Y_avg, "x_noiseless_simulation": x_ns, "y_noiseless_simulation": y_ns,
      "x_real_hardware": x_rh, "y_real_hardware": y_rh, "x_noiseless_simulation_loss": x_nsl, "y_noiseless_simulation_loss": y_nsl,
      "x_real_hardware_loss": x_rhl, "y_real_hardware_loss": y_rhl}
df = pd.DataFrame(dd)

df.to_csv(os.path.basename(__file__) + ".plot_data_dump.csv")
plt.plot(X_padded[0], Y_avg, color='gray', label='Random*', linestyle='dashed', linewidth=2)
plt.ylabel('Cut')
plt.xlabel('# Iterations')
plt.grid(visible=True)

leg = plt.legend(loc='lower right')
plt.xlim([0,45])
# plt.savefig("avg_hw_and_sim_and_random_and_optimal_8_nodes.png", bbox_inches='tight',figsize=(8, 8), dpi=120)
# 8 nodes

# '129785', '46'
# '129786', '251'
# '129787', '251'
# '129854', '42'
# '129855', '56'
# '129856', '42'


plt.show()

plt.show()

