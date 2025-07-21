# insert imports
import argparse
import base64
import zlib

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
import json
import networkx as nx
import math
from DbAdapter import DbAdapter
import time
from qiskit import IBMQ


def maxcut_global(args):
    dbAdapter = DbAdapter()
    stepsize = args.stepsize
    steps = args.steps
    shots = args.shots
    backend = args.backend
    comment = args.comment
    expected_maxcut = args.expected_maxcut
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-08
    print(f"stepsize:{stepsize}")

    expected_b = args.expected_b
    fileName = args.fileName
    layers = int(args.layers)
    g = nx.read_graph6(fileName)

    nodes = list(g.nodes)
    graph = list(g.edges)
    log2Nodes = math.log2(len(nodes))
    if log2Nodes.is_integer():
        n_wires = int(log2Nodes)
    else:
        n_wires = int(math.ceil(log2Nodes))
    # n_wires = int(math.log2(len(nodes)))
    steps = steps  # wire_num_steps_dic[n_wires]

    # dev = qml.device('default.qubit', wires=n_wires)
    #ibm-q-research-2/bar-ilan-uni-1/main
    provider = IBMQ.enable_account(
        "4c76f1fa873aa6cf754a86bb5d7189c2d98ca02a6b51b0bd08a617243428164e4eb2d079bb2962a023bdf7033353795d950cdf26efff19ae4fcdd9911b0f2506",
        hub='ibm-q-research-2', group='bar-ilan-uni-1', project='main')
    # provider = IBMQ.enable_account(
    #     "fb947e2303b279891b3c0c72d0480fb40035b89a3b8138f3596280199d6cbea311c3d83cea46096d9bb99d81107d5fbf6890ea9c2367f12a8ea83970b05cf1fb",
    #     hub='ibm-q-research-2', group='bar-ilan-uni-1', project='main')


    dev = qml.device('qiskit.ibmq', wires=n_wires, backend=backend,
                     provider=provider, shots=shots)
    print(dev.capabilities()['backend'])
    @qml.qnode(dev)
    def circuit(params):
        for index in range(n_wires):
            qml.Hadamard(index)
        StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.probs(wires=range(n_wires))

    # init_weights = qml.init.strong_ent_layers_uniform(n_layers=layers, n_wires=n_wires)
    init_weights = np.random.random(StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_wires))
    json_str = json.dumps(init_weights.numpy().tolist())
    dbAdapter.new_experiment(fileName, expected_b, len(nodes), layers, True, steps, json_str, comment, stepsize, beta1, beta2, eps, shots=shots, backend=backend)

    params = init_weights
    print(init_weights)
    print(circuit(init_weights))

    # print(circuit.draw())

    vertex_num = 2 ** n_wires

    print(vertex_num)

    def cost_reversed(probs_results):
        num_blacks = expected_b
        total_cost = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]
            total_cost += (abs(probs_results[vertex1] + probs_results[vertex2]) - (1 / (num_blacks))) ** 2

        return total_cost

    def cost_only(prob_results):
        binary_results = probs2binary(probs_results)
        cost_val_old = cost_per_assignment(probs_results)
        #  print(cost_val_old)
        cost_val = cost_per_assignment(probs_results) + cost_reversed(probs_results)

        #    print(cost_val)
        cost_from_clean_solution = cost_from_clean_sol(probs_results)
        #    print(cost_from_clean_solution)

        cut_value = cut(binary_results)

        # print("Cut = {:5d}".format(cut_value))

        return cost_val
    def cost(params):
        probs_results = circuit(params)
        binary_results = probs2binary(probs_results)
        cost_val_old = cost_per_assignment(probs_results)
      #  print(cost_val_old)
        cost_val = cost_per_assignment(probs_results) + cost_reversed(probs_results)

    #    print(cost_val)
        cost_from_clean_solution = cost_from_clean_sol(probs_results)
    #    print(cost_from_clean_solution)

        cut_value = cut(binary_results)

        # print("Cut = {:5d}".format(cut_value))

        return cost_val

    def probs2binary(probs_results):
        binary_results = [1 if prob > 1 / (2 * expected_b) else 0 for prob in probs_results]
        return binary_results

    def cost_per_assignment(probs_results):
        total_cost = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]
            total_cost += (abs(probs_results[vertex1] - probs_results[vertex2]) - (1 / expected_b)) ** 2
        return total_cost

    def cost_from_clean_sol(probs_results):
        total_cost = 0
        for prob in probs_results:
            if (prob < 1 / (2 * expected_b)):
                total_cost += prob ** 2
            else:
                total_cost += (prob - (1 / expected_b)) ** 2
        return total_cost

    def cut(binary_results):
        total_cut = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]
            if (binary_results[vertex1] != binary_results[vertex2]):
                total_cut += 1
        return total_cut

    print(cost(params))

    opt = qml.AdamOptimizer(stepsize=stepsize, beta1=beta1, beta2=beta2, eps=eps)  # find the solution but does not converge to it
    # set the initial parameter values
    params = init_weights
    print(params)

    step_jumps = 1

    print("steps:", steps)
    x0 = np.arange(start=0, stop=steps + 1, step=step_jumps)

    x = x0
    # print(x0)

    # print(x)
    # print(len(x))

    costs_arr = np.zeros(len(x))
    cuts_arr = np.zeros(len(x))
    first_max_maxcut_reached = False
    j = 0
    end_counter = 0;
    # print(params)
    max_cut = 0
    max_cut_found_at = 0
    for i in range(steps+1):
        if i % step_jumps == 0:
            dbAdapter.add_iteration(i)
            costs_arr[j] = cost(params)
            probs_results = circuit(params)
            dbAdapter.add_probs(str(probs_results))
            # print("probs_results:" + str(probs_results))
            binary_results = probs2binary(probs_results)
            cuts_arr[j] = cut(binary_results)

            if cuts_arr[j] > max_cut:
                max_cut = cuts_arr[j]
                max_cut_found_at = i
            if expected_maxcut == max_cut and not first_max_maxcut_reached:
                first_max_maxcut_reached = True
            if first_max_maxcut_reached:
                end_counter += 1
                if 0.2 * steps < end_counter:
                    break
           # print('j=', j)
           # print('i=', i)
            dbAdapter.add_loss(np.asscalar(costs_arr[j]))
            #print("Cost after step {:5d}: {: .7f}".format(i, costs_arr[j]))
            dbAdapter.add_group(str(binary_results))
            dbAdapter.add_blacks(binary_results.count(1))
            # print(str(probs2binary(probs_results)))
            dbAdapter.add_muxcut(float(str(max_cut)))
            print('max_cut=', max_cut)
            print('max_cut found at =', max_cut_found_at)
            #print("Optimized rotation angles: {}".format(params))
            j += 1

        # update the circuit parameters after each step relative to cost & params
        L_steps=1
        for rounds in range(L_steps):
            params = opt.step(lambda v: cost(v), params)


    #print(cuts_arr)


parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("expected_b", type=int)
parser.add_argument("fileName", type=str)
parser.add_argument("layers", type=str)
parser.add_argument("stepsize", type=float)
parser.add_argument("steps", type=int)
parser.add_argument("shots", type=int)
parser.add_argument("backend", type=str)
parser.add_argument("comment", type=str)
parser.add_argument("expected_maxcut", type=int)
args = parser.parse_args()

maxcut_global(args)
