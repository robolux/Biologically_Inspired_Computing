# Project 1 Testing - Computational Physics
# Hunter Phillips

import sys
from movements import data_call
import mlp
import os.path
import numpy as nmp
import math
import time

filename = 'data/movements_day1-3.dat'
train, train_targets, valid, valid_targets, test, test_targets = data_call(filename=filename)

save_path = '../results'

for w, g in enumerate(['4', '6', '8', '12'], 1): # number of hidden nodes
    for w2, g2 in enumerate(['10', '100', '1000'], 1):
        save_path = '../results/'
        filename = os.path.join(save_path, "n_" + str(g) + "_i_" + str(g2) + ".txt")
        print(filename)
        with open (filename, 'w') as f_m:           # with auto closes files which is helpful here

            hidn_i      = int(g)
            iterations  = int(g2)
            eta_i       = 0.1
            beta_i      = 1.0
            bias_i      = -1.0
            momentum_i  = 0.0

            start_time = time.time()
            active = mlp.mlp(train, train_targets, hidn_i, beta = beta_i, eta = eta_i, bias = bias_i, momentum = momentum_i)
            active.earlystopping(train, train_targets, valid, valid_targets, iterations)
            end_time = time.time()
            general_time = end_time - start_time

            c_matrix = active.confusion(test, test_targets)

            f_m.write('\nResulting Confusion Matrix\n')
            f_m.write("************************************\n")
            nmp.savetxt(f_m,c_matrix, fmt = '%2.0f')

            # Output Class Prediction Percentages
            f_m.write('\n\nClass Prediction Percentages\n')
            f_m.write("************************************")
            average = 0
            for m, n in enumerate(c_matrix.transpose()):
            	value = (n[m]/nmp.sum(n))*100
            	if math.isnan(value):
            		value = 0

            	f_m.write("\n{} : {}%".format(m+1,value))
            	average+=value

            # Output Average Percentage
            f_m.write('\nThe average percentage is {}%\n'.format(average/8))

            # Benchmark Times
            f_m.write("\nThe Elapsed Computational Time: {} seconds".format(general_time))
