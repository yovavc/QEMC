import os
from chardet import detect
import re
import matplotlib.pyplot as plt
def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']

directory = "groupBy\\same_number_of_size_and_layers\\16_2\\ones_3"

files = os.listdir(directory)
x_all = []
y_all = []
files.sort()
graph_name = None
for file in files:
    print(file.split(".")[0])
    if graph_name != file.split(".")[0]:

        if len(x_all) > 0:
            average_max_maxcut = 0
            plt.ylabel('MaxCut')
            plt.xlabel('Iterations')
            plt.title(file + "\n" + graph_name)
            plt.grid(visible=True)

            for i in range(len(x_all)):
                if len(y_all[i]) > 0:
                    plt.plot(x_all[i], y_all[i])
                    average_max_maxcut += y_all[i][-1]
            average_max_maxcut = average_max_maxcut / len(x_all)
            plt.axhline(y=average_max_maxcut, color='r', linestyle='--')
            plt.text(1000, average_max_maxcut, str(average_max_maxcut))
           #plt.savefig(directory + "_plots/" + graph_name + ".plt" + ".png")
            plt.show()
            plt.clf()
            x_all = []
            y_all = []
    graph_name = file.split(".")[0]
    # writing to file
    file1 = open(directory + "/" + file, 'r', encoding='utf-8', errors='ignore')
    count = 0
    Lines = file1.readlines()
    print(file)
    file1.close()
    step = None
    cut = None
    max_cut_found_at = None
    latest_max_cut = 0
    x = []
    y = []
    for line in Lines:
        strippedLine = line.strip()
        if re.search("max_cut=", strippedLine):
            cut = float(re.sub(' +', ' ', strippedLine).split(" ")[1])
        if re.search("after step", strippedLine):
            step = int(re.sub(' +', ' ', strippedLine.split(":")[0]).split(" ")[3])
        if re.search("max_cut found at =", strippedLine):
            max_cut_found_at = int(re.sub(' +', ' ', strippedLine).split(" ")[4])
        if cut is not None and step is not None and max_cut_found_at is not None:
            x.append(step)
            y.append(cut)
            latest_max_cut = max_cut_found_at
            # print(cut, step, max_cut_found_at)
            cut = None
            step = None
            max_cut_found_at = None
    #x_all.append(x[0:int(latest_max_cut * 1.1)])
    #y_all.append(y[0:int(latest_max_cut * 1.1)])
    x_all.append(x)
    y_all.append(y)
    plt.ylabel('MaxCut')
    plt.xlabel('Iterations')
    plt.title(file + "\n" + graph_name)
    plt.grid(visible=True)
    plt.plot(x_all[-1], y_all[-1])
    plt.show()
    #plt.savefig(directory + "_plots/" + file + ".plt" + ".png")
    plt.clf()







