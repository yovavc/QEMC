#!/usr/bin/env python

# Copyright 2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Goemans-Williamson classical algorithm for MaxCut
"""

from typing import Tuple

import cvxpy as cvx
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from quantumflow.utils import from_graph6


def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """
    # Kudos: Originally implementation by Nick Rubin, with refactoring and
    # cleanup by Jonathon Ward and Gavin E. Crooks
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # Setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)
    obj = cvx.Maximize(cvx.trace(laplacian * psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T

    # Bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)

    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector /= np.linalg.norm(random_vector)
    colors = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = colors @ laplacian @ colors.T

    return colors, score, bound


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


if __name__ == '__main__':
    # Quick test of GW code
 #   for i in range(0, 1000):
    G = from_graph6(r"My]WObEnkmHl}i}\_")    # 14 nodes
    g = create_graph(2**3, 2)
    nx.write_graph6(g, 'anedge.g6')
    g1 = nx.read_graph6('anedge.g6')
    nx.draw(g1)
    plt.show()
    laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
    bound = goemans_williamson(G)[2]
    assert np.isclose(bound, 36.25438489966327)
    print(goemans_williamson(G))
    scores = [goemans_williamson(G)[1] for n in range(100)]
    assert max(scores) >= 34
    print(min(scores), max(scores))
    exit()

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    nx.write_graph6(g, 'anedge.g6')
    g1 = nx.read_graph6('anedge.g6')

    nx.draw(g1)
    plt.show()
    laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
    bound = goemans_williamson(G)[2]

    assert np.isclose(bound, 36.25438489966327)
    print(goemans_williamson(G))

    scores = [goemans_williamson(G)[1] for n in range(100)]
    assert max(scores) >= 34

    print(min(scores), max(scores))