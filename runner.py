import os
for j in range(1, 7):
    for i in range(1, 10):
        os.system("python no_ent_circular.py " + str(j) + " > no_ent_out" + str(j) + "_" + str(i) + ".txt")

for j in range(1, 7):
    for i in range(1, 10):
        os.system("python with_ent_circular.py " + str(j) + " > with_ent_out" + str(j) + "_" + str(i) + ".txt")
