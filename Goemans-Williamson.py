#!/usr/bin/env python

# Copyright 2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Goemans-Williamson classical algorithm for MaxCut
"""
import argparse
from typing import Tuple

import cvxpy as cvx
import networkx as nx
import numpy as np

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


parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)

if __name__ == '__main__':
    # Quick test of GW code
    args = parser.parse_args()
    G = nx.read_graph6(args.fileName)
    scores = []
    bounds = []
    colors = []
    for n in range(100):
        color, score, bound = goemans_williamson(G)
        colors.append(color)
        bounds.append(bound)
        scores.append(score)
    max_color = colors[scores.index(max(scores))]
    print("minScore:"+ str(min(scores))+",maxScore:" + str(max(scores)) + ",maxColor:" + str(max_color)
          +",group0Count:" + str(np.count_nonzero(max_color == -1)) + ",group1Count:"
          + str(np.count_nonzero(max_color == 1)))
