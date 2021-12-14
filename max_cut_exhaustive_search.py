import argparse
import networkx as nx

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
args = parser.parse_args()


def get_max_cut(graph6):
    graph = nx.read_graph6(args.fileName)
    edges = list(graph.edges)
    nodes = graph.nodes
    global_max_cut = 0
    global_max_cut_combination = 0
    for i in range(0, 2 ** (len(nodes))):
        max_cut_for_i = 0
        for edge in edges:
            vertex_1 = 1 << edge[0]
            vertex_2 = 1 << edge[1]
            mask = (1 << edge[1]) + (1 << edge[0])
            active_bits = i & mask
            is_arc_between_two_groups = (active_bits & (active_bits-1) == 0) and active_bits != 0
            print(f'{mask:032b}' + " " + f'{i:032b}' + " " + f'{active_bits:032b}' + " " + str(is_arc_between_two_groups))
            if is_arc_between_two_groups:
                max_cut_for_i += 1
        print("current max cut: " + str(max_cut_for_i) + " for combination " + str(i) + " " + bin(i))
        if global_max_cut< max_cut_for_i:
            global_max_cut = max_cut_for_i
            global_max_cut_combination = i

    print("global max cut: " + str(global_max_cut) + " for combination " + str(global_max_cut_combination)
          + " " + bin(global_max_cut_combination))
    print(edges)


get_max_cut(args.fileName)
