# insert imports
import argparse
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.init import strong_ent_layers_uniform
from pennylane.templates import SimplifiedTwoDesign
from pennylane.init import (simplified_two_design_initial_layer_normal,
                            simplified_two_design_weights_normal)

import networkx as nx
import matplotlib.pyplot as plt

import random

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("wires", type=int)
args = parser.parse_args()

n_wires = args.wires
expected_b = 2 ** (n_wires-1)
dev = qml.device('default.qubit', wires=n_wires)

@qml.qnode(dev)
def circuit(params):
    for index in range(n_wires):
        qml.Hadamard(index)
    StronglyEntanglingLayers(params, wires=list(range(n_wires)))
    return qml.probs(wires=range(n_wires))


init_weights = strong_ent_layers_uniform(n_layers=2, n_wires=n_wires)
params = init_weights
print(init_weights)
print(circuit(init_weights))
#print(circuit(state=3))
print(circuit.draw())

vertex_num = 2 ** n_wires

print(vertex_num)

list_a = list(range(0, vertex_num))
list_b = list(range(1, vertex_num)) + [0]
graph = list(zip(list_a, list_b))

params = init_weights
print(params)


def cost(params):
    probs_results = circuit(params)
    # print(probs_results)
    binary_results = probs2binary(probs_results)
    # print(binary_results)
    cost_val = cost_per_assignment(probs_results)
    print(cost_val)
    cost_from_clean_solution = cost_from_clean_sol(probs_results)
    print(cost_from_clean_solution)
    # cost_val += cost_from_clean_solution
    # print(cost_val)
    cut_value = cut(binary_results)
    # cost_val+=(16-cut_value)/500

    # print("Cut = {:5d}".format(cut(binary_results)))
    print("Cut = {:5d}".format(cut_value))
    print("cost:" + str(cost_val))
    return cost_val


def probs2binary(probs_results):
    binary_results = probs_results
    # binary_results = [1 if prob >= 2/vertex_num else 0 for prob in probs_results]
    binary_results = [1 if prob > 1 / (2*expected_b) else 0 for prob in probs_results]
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


def cost_per_assignment(probs_results):
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
        if prob < 1 / (2*expected_b):
            total_cost += prob ** 2
        else:
            total_cost += (prob - (1 / expected_b)) ** 2
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
# opt = qml.GradientDescentOptimizer(stepsize=0.9)
# opt = qml.GradientDescentOptimizer(stepsize=0.95)
# opt = qml.AdamOptimizer(stepsize=0.4, beta1=0.9, beta2=0.99, eps=1e-08)
# opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.3, beta2=0.3, eps=1e-08)
opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.1, eps=1e-08)  # find the solution but does not converge to it
# opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.7, beta2=0.1, eps=1e-08) # find the solution but does not converge to it
# opt = qml.AdagradOptimizer(stepsize=0.05, eps=1e-08)
# set the number of steps
steps = 50000
# set the initial parameter values
# params = np.random.uniform(size=(num_layers, num_helper_qubits*2, 3))
params = init_weights
print(params)

step_jumps = 1

print("total steps:" + str(steps))
x0 = np.arange(start=0, stop=steps + 1, step=step_jumps)
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
for i in range(steps + 1):
        if (i) % step_jumps == 0:
                costs_arr[j] = cost(params)
                probs_results = circuit(params)
                binary_results = probs2binary(probs_results)
                cuts_arr[j] = cut(binary_results)
                if (cuts_arr[j] > max_cut):
                        max_cut = cuts_arr[j]
                        max_cut_found_at = i
                print('j=', j)
                print('i=', i)
                print("Cost after step {:5d}: {: .7f}".format(i, costs_arr[j]))
                print(probs2binary(probs_results))
                print('max_cut=', max_cut)
                print('max_cut found at =', max_cut_found_at)
                # print("Optimized rotation angles: {}".format(params))
                j += 1
                if ((max_cut == vertex_num) and (i > 1.3 * max_cut_found_at)):
                        break
        # update the circuit parameters after each step relative to cost & params
        L_steps = 1
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

# plt.plot(x, costs_arr, 'b', linewidth=2.5)
# plt.ylim(ymin=ymin_fixed, ymax=0.3)
# plt.ylabel(r'$Cost$', fontsize=14)
# # plt.yscale("log")
# plt.xlabel('# steps', fontsize=16)
#
# x_max = 1.3 * max_cut_found_at
# # plt.xlim(xmin=0, xmax=x_max)
# plt.subplots_adjust(top=1, bottom=0.4, left=0.10, right=1.5, hspace=0.4,
#                     wspace=0.35)
#
# # plt.show()
#
# plt.plot(x, cuts_arr, 'b', linewidth=2.5)
# plt.ylim(ymin=0, ymax=vertex_num + 0.5)
# plt.ylabel(r'$Cost$', fontsize=14)
# # plt.yscale("log")
# plt.xlabel('# steps', fontsize=16)
#
# plt.xlim(xmin=0, xmax=x_max)
# plt.subplots_adjust(top=1, bottom=0.4, left=0.10, right=1.5, hspace=0.4,
#                     wspace=0.35)

# plt.show()
