import numpy as np
import networkx as nx
import random
import argparse

def create_graph(nodes, node_number_of_connection):
    # Creating a node list
    node_list = list(range(0, nodes))

    # creating graph object
    maxcut_graph = nx.Graph()

    # here we create a random graph, but since the graph might not be connected we keep trying to create graph until
    # we get a connected graph
    while(True):
        # Connecting node to other "node_number_of_connection" random nodes
        for node in node_list:
            # Creating a copy list and removing the subject node so it wont connect to himself
            temp_node_list = node_list.copy()
            temp_node_list.remove(node)
            for j in range(0, node_number_of_connection):

                chosen_node = random.choice(temp_node_list)
                # Creating a random edge
                maxcut_graph.add_edge(node, chosen_node)

                # removing chosen node so they wont get selected twice
                temp_node_list.remove(chosen_node)

        # Checking if the graph is connect (no ilands in the graph)
        if nx.is_connected(maxcut_graph):
            print("graph not connected, trying again")
            break
    return maxcut_graph


def create_graph_by_article(nodes):
    # Creating a node list
    node_list = list(range(0, nodes))

    # creating graph object
    maxcut_graph = nx.Graph()

    # here we create a random graph, but since the graph might not be connected we keep trying to create graph until
    # we get a connected graph
    while(True):
        # Connecting node to other "node_number_of_connection" random nodes
        for node in node_list:
            # Creating a copy list and removing the subject node so it wont connect to himself
            temp_node_list = node_list.copy()
            temp_node_list.remove(node)
            for j in range(0, len(temp_node_list)):
                if random.uniform(0, 1) > 0.5:
                    maxcut_graph.add_edge(node, temp_node_list[j])

        # Checking if the graph is connect (no ilands in the graph)
        if nx.is_connected(maxcut_graph):
            print("graph not connected, trying again")
            break
    return maxcut_graph


parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("count", type=int)
parser.add_argument("size", type=int, help="graph is 2 in the power of size")
args = parser.parse_args()

for i in range(args.count):
    graph = create_graph_by_article(args.size)
    string = "vertices_" + str(args.size) + "_index_" + str(i) + ".g6"
    nx.write_graph6(graph, string)

