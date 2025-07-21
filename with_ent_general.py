# insert imports
import argparse
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
import MySQLdb
from pennylane.init import strong_ent_layers_uniform
# from pennylane.templates import SimplifiedTwoDesign
# from pennylane.init import (simplified_two_design_initial_layer_normal,
#                             simplified_two_design_weights_normal)

import networkx as nx
import matplotlib.pyplot as plt

import random
import math

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("expected_b", type=int)
parser.add_argument("fileName", type=str)
parser.add_argument("layers", type=str)
args = parser.parse_args()


def maxcut_global(args):
    expected_b = args.expected_b
    fileName = args.fileName
    layers = int(args.layers)
    g = nx.read_graph6(fileName)
    nodes = list(g.nodes)
    graph = list(g.edges)
    n_wires = int(math.log2(len(nodes)))

    dev = qml.device('default.qubit', wires=n_wires)
    # @qml.qnode(dev)
    # def circuit(params):
    #     for index in range(n_wires):
    #         qml.Hadamard(index)
    #     for index in range(n_wires):
    #         qml.Rot(params[0][index][0], params[0][index][1], params[0][index][2], wires=index)
    #     return qml.probs(wires=range(n_wires))
    @qml.qnode(dev)
    def circuit(params):
        for index in range(n_wires):
            qml.Hadamard(index)
        StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.probs(wires=range(n_wires))

    init_weights = strong_ent_layers_uniform(n_layers=layers, n_wires=n_wires)
    #init_weights = [[[3.14002626, 1.18485963, 4.52794101],[0.37923013, 2.14754543, 3.17863271]],[[6.03876067, 3.75811232, 4.08626339],[3.70709778, 0.97374692, 1.25603593]]]
    print("init_weights:" + str())
    params = init_weights
    print(init_weights)
    print(circuit(init_weights))
    #print(circuit(state=3))
    print(circuit.draw())

    vertex_num = 2 ** n_wires

    print(vertex_num)

    # list_a = list(range(0, vertex_num))
    # list_b = list(range(1, vertex_num)) + [0]
    # graph = list(zip(list_a, list_b))
    def cost_reversed(probs_results):
        num_blacks = 128
        total_cost = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]

            # total_cost += (abs(probs_results[vertex1]-probs_results[vertex2])-(2/(vertex_num)))**2
            total_cost += (abs(probs_results[vertex1] + probs_results[vertex2]) - (1 / (num_blacks))) ** 2

        return total_cost

    def cost(params):
        probs_results = circuit(params)
        # print(probs_results)
        binary_results = probs2binary(probs_results)
        # print(binary_results)
        cost_val_old = cost_per_assignment(probs_results)
        print(cost_val_old)
        # cost_val = cost_per_assignment(probs_results)+3*cost_from_clean_sol(probs_results)
        cost_val = cost_per_assignment(probs_results) + cost_reversed(probs_results)
        print(cost_val)
        cost_from_clean_solution = cost_from_clean_sol(probs_results)
        print(cost_from_clean_solution)
        # cost_val += cost_from_clean_solution
        # print(cost_val)
        cut_value = cut(binary_results)
        # cost_val+=(16-cut_value)/5008
        # print("Cut = {:5d}".format(cut(binary_results)))
        print("Cut = {:5d}".format(cut_value))

        return cost_val


    def probs2binary(probs_results):
        binary_results = probs_results
        # binary_results = [1 if prob >= 2/vertex_num else 0 for prob in probs_results]
        binary_results = [1 if prob > 1/(2 * expected_b) else 0 for prob in probs_results]
        # binary_results = [round (prob+ 1/vertex_num) for prob in probs_results]

        #     new_prices = [round(price - (price * 10 / 100), 2) if price > 50 else price for price in prices]
        #     print(new_prices)
        #     for index, prob in enumerate(probs_results):
        #         print(1/vertex_num)
        #         if (prob > 1/vertex_num):
        #             binary_results[index] = 1
        #         else:
        #             binary_results[index] = -1
        return binary_results

    def cost_per_assignment(  probs_results):
        total_cost = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]
            # total_cost += (2*binary_results[vertex1]-1)*(2*binary_results[vertex2]-1)
            # Note that we enter probs_results as input and not binary_results
            # total_cost += (abs(binary_results[vertex1]-binary_results[vertex2])-0.25)**2
            # total_cost += (abs(binary_results[vertex1]-binary_results[vertex2])-1)**2
            total_cost += (abs(probs_results[vertex1] - probs_results[vertex2]) - (1 / expected_b)) ** 2
            # total_cost += abs(abs(probs_results[vertex1]-probs_results[vertex2])-(2/(vertex_num)))
            # total_cost += ((probs_results[vertex1]-probs_results[vertex2])**2-(2/(vertex_num))**2)**2
            # total_cost += ((probs_results[vertex1]-probs_results[vertex2])**2-(2/(vertex_num))**2)**2
            # total_cost += ((10*(probs_results[vertex1]-probs_results[vertex2]))**2-(20/(vertex_num))**2)**2
        return total_cost


    def cost_from_clean_sol(probs_results):
        total_cost = 0
        for prob in probs_results:
            if (prob < 1/(2 * expected_b)):
                total_cost += prob ** 2
            else:
                total_cost += (prob - (1/expected_b)) ** 2
        return total_cost


    def cut(binary_results):
        total_cut = 0
        for edge in graph:
            vertex1 = edge[0]
            vertex2 = edge[1]
            # print(vertex1)
            # print(vertex2)
            # print (binary_results[vertex1])
            # print (binary_results[vertex2])
            if (binary_results[vertex1] != binary_results[vertex2]):
                total_cut += 1
        return total_cut


    print(cost(params))
    # print(total_cost(params))
    # print(total_cost_uniform(params))

    # initialise the optimizer with stepsize
    #opt = qml.GradientDescentOptimizer(stepsize=0.9)
    #opt = qml.GradientDescentOptimizer(stepsize=0.95)
    #opt = qml.AdamOptimizer(stepsize=0.4, beta1=0.9, beta2=0.99, eps=1e-08)
    #opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.3, beta2=0.3, eps=1e-08)
    opt = qml.AdamOptimizer(stepsize=0.2, beta1=0.9, beta2=0.99, eps=1e-08) # find the solution but does not converge to it
    #opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.7, beta2=0.1, eps=1e-08) # find the solution but does not converge to it
    #opt = qml.AdagradOptimizer(stepsize=0.05, eps=1e-08)
    # set the number of steps
    wire_num_steps_dic = {
        1: 100,
        2: 3000,
        3: 10000,
        4: 25000,
        5: 50000,
        6: 100000
    }
    steps = 200 #wire_num_steps_dic[n_wires]

    # set the initial parameter values
    #params = np.random.uniform(size=(num_layers, num_helper_qubits*2, 3))
    params = init_weights
    print(params)

    step_jumps = 1

    print("steps:", steps)
    x0 = np.arange(start=0, stop=steps+1, step=step_jumps)
    ###x1 = np.arange(start = steps+1, stop = 2*(steps+1), step=step_jumps)

    ###x = np.concatenate((x0,x1,x2,x3,x4),0)
    x = x0
    print(x0)
    ###print(x1)
    print(x)
    print(len(x))
    meas_res_arr = np.zeros(len(x))

    meas_res_arr0 = np.zeros(len(x))
    meas_res_arr1 = np.zeros(len(x))
    meas_res_arr2 = np.zeros(len(x))
    meas_res_arr3 = np.zeros(len(x))

    meas_res_err_arr = np.zeros(len(x))
    meas_res_err_arr0 = np.zeros(len(x))
    meas_res_err_arr1 = np.zeros(len(x))
    meas_res_err_arr2 = np.zeros(len(x))
    meas_res_err_arr3 = np.zeros(len(x))
    meas_res_err_bitflip_arr = np.zeros(len(x))
    costs_arr = np.zeros(len(x))
    cuts_arr = np.zeros(len(x))
    angle_arr = np.zeros(len(x))
    opt_var_arr = np.zeros(len(x))

    j = 0
    print(params)
    max_cut = 0
    max_cut_found_at = 0
    for i in range(steps+1):
        if (i) % step_jumps == 0:
            costs_arr[j] = cost(params)
            probs_results = circuit(params)
            print("probs_results:" + str(probs_results))
            binary_results = probs2binary(probs_results)
            cuts_arr[j] = cut(binary_results)
            if (cuts_arr[j]> max_cut):
                max_cut = cuts_arr[j]
                max_cut_found_at = i
            print('j=',j)
            print('i=',i)
            print("Cost after step {:5d}: {: .7f}".format(i, costs_arr[j]))
            print(probs2binary(probs_results))
            print('max_cut=',max_cut)
            print('max_cut found at =',max_cut_found_at)
            #print("Optimized rotation angles: {}".format(params))
            j+=1

        # update the circuit parameters after each step relative to cost & params
        L_steps=1
        for rounds in range(L_steps):
            params = opt.step(lambda v: cost(v), params)

    # #print("Optimized rotation angles: {}".format(params))

    # ymin_fixed=0.00001
    # vertex_num=32
    # plt.plot(x,costs_arr,'b',linewidth=2.5)
    # plt.ylim(ymin=ymin_fixed,ymax=0.3)
    # plt.ylabel(r'$Cost$',fontsize=14)
    # #plt.yscale("log")
    # plt.xlabel('# steps',fontsize=16)

    # plt.xlim(xmin=0,xmax=len(x)-1)
    # plt.subplots_adjust(top=1, bottom=0.4, left=0.10, right=1.5, hspace=0.4,
    #                     wspace=0.35)

    # plt.show()


    # #print("Optimized rotation angles: {}".format(params))

    ymin_fixed = 0.00001

    plt.plot(x, costs_arr, 'b', linewidth=2.5)
    plt.ylim(ymin=ymin_fixed, ymax=0.3)
    plt.ylabel(r'$Cost$', fontsize=14)
    # plt.yscale("log")
    plt.xlabel('# steps', fontsize=16)

    x_max = 1.3 * max_cut_found_at
    # plt.xlim(xmin=0, xmax=x_max)
    # plt.subplots_adjust(top=1, bottom=0.4, left=0.10, right=1.5, hspace=0.4,
    #                     wspace=0.35)

    #plt.show()

    plt.plot(x, cuts_arr, 'b', linewidth=2.5)
    plt.ylim(ymin=0, ymax=vertex_num + 0.5)
    plt.ylabel(r'$Cost$', fontsize=14)
    # plt.yscale("log")
    plt.xlabel('# steps', fontsize=16)

    # plt.xlim(xmin=0, xmax=x_max)
    # plt.subplots_adjust(top=1, bottom=0.4, left=0.10, right=1.5, hspace=0.4,
    #                     wspace=0.35)

    # plt.show()

    print(cuts_arr)


maxcut_global(args)