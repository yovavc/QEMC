import argparse

import cvxpy as cvx
import networkx as nx
import numpy as np

from numba import jit

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import time
def split(cut):
    n = len(cut)
    S = [i for i in range(n) if cut[i]]
    T = [i for i in range(n) if not cut[i]]
    return S, T


def cut_cost1(cut, G):
    S, T = split(cut)
    l = list(nx.edge_boundary(G, S, T))
    return len(l)


def cut_cost(x, L):
    return 0.25 * x @ L @ x


def int_to_binary(n, int_cut):
    """Converts bitmask(==int) cut representation to list of bits"""

    return np.array([int(c) for c in bin(int_cut)[2:].zfill(n)])


def brute_force_max_cut(G):
    """Compute maximum cut of a graph considering all the possible cuts."""""

    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes))

    max_cut_value = 0
    max_cut = 0

    for int_cut in range(1, 2**(n-1) + 1):
        cut = int_to_binary(n, int_cut)
        value = cut_cost(cut * 2 - 1, L)

        if value > max_cut_value:
            max_cut_value = value
            max_cut = cut

    return max_cut_value


def SDP_max_cut(G, num_trials):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes))

    # SDP solution
    X = cvx.Variable((n, n), PSD=True)
    obj = 0.25 * cvx.trace(L.toarray() * X)
    constr = [cvx.diag(X) == 1]
    problem = cvx.Problem(cvx.Maximize(obj), constraints=constr)
    problem.solve(solver=cvx.SCS)

    # GW algorithm
    u, s, v = np.linalg.svd(X.value)
    U = u * np.sqrt(s)

    # num_trials = 10 # УЗБЧ
    gw_results = np.zeros(num_trials)
    for i in range(num_trials):
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)
        cut = np.sign(r @ U.T)
        print(cut)
        print(sum(cut))
        print((n-sum(cut))/2)
        gw_results[i] = cut_cost(cut, L)
        print(gw_results[i])

    # Verbose result
    _ = plt.hist(gw_results, bins=100)
    return (np.mean(gw_results), np.max(gw_results))


parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("fileName", type=str)
parser.add_argument("num_trials", type=int)


args = parser.parse_args()
G = nx.read_graph6(args.fileName)
num_trials = args.num_trials
start_time = time.time()
mean_gw, max_cut_gw = SDP_max_cut(G, num_trials)
end_time = time.time()

lap_time = round((end_time - start_time), 4)
print("mean:" + str(mean_gw), "max_cut_gw:" + str(max_cut_gw), "time:" + str(lap_time))

