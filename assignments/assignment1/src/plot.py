# Assignment 1 Plotting - Biologically Inspired Computing
# Hunter Phillips

import matplotlib.pyplot as plt
from solvers import *
import numpy as nmp
import sys
import os.path

save_path = '../results/plots'


# 10 Cities

cities, distance = open_file() # open file
cs, cl = num_cities(10,distance) # reshape with requested size to compute

var_hold, nr_bt = GA(cities, distance, cs, cl, 10, 100, 0.2, 20, 10, 10)
av_ga = nmp.mean(var_hold, axis=0)
x = nmp.linspace(1., len(av_ga), len(av_ga))
plt.plot(x, av_ga, 'r', label = 'Population of 10')

var_hold, nr_bt = GA(cities, distance, cs, cl, 40, 100, 0.2, 20, 10, 10)
av_ga = nmp.mean(var_hold, axis=0)
plt.plot(x, av_ga, 'y', label = 'Population of 30')

var_hold, nr_bt = GA(cities, distance, cs, cl, 160, 100, 0.2, 20, 10, 10)
av_ga = nmp.mean(var_hold, axis=0)
plt.plot(x, av_ga, 'g', label = 'Population of 90')

plt.xlabel('Generation')
plt.legend(loc='upper right')
plt.ylabel('Distance')
plt.title('Fitness of Genetic Algorithm with 10 Cities')
filename = os.path.join(save_path,"10_cities.png")
plt.savefig(filename)   # save figure to .png in ../results/plots
plt.clf()               # clear figure to prepare for next algo plot


# 24 Cities

cities, distance = open_file()      # open file
cs, cl = num_cities(24,distance)    # reshape with requested size to compute

var_hold, nr_bt = GA(cities, distance, cs, cl, 10, 100, 0.2, 20, 10, 24)
av_ga = nmp.mean(var_hold, axis=0)
x = nmp.linspace(1., len(av_ga), len(av_ga))
plt.plot(x, av_ga, 'r', label = 'Population of 10')

var_hold, nr_bt = GA(cities, distance, cs, cl, 40, 100, 0.2, 20, 10, 24)
av_ga = nmp.mean(var_hold, axis=0)
plt.plot(x, av_ga, 'y', label = 'Population of 30')

var_hold, nr_bt = GA(cities, distance, cs, cl, 160, 100, 0.2, 20, 10, 24)
av_ga = nmp.mean(var_hold, axis=0)
plt.plot(x, av_ga, 'g', label = 'Population of 90')

plt.xlabel('Generation')
plt.legend(loc='upper right')
plt.ylabel('Distance')
plt.title('Fitness of Genetic Algorithm with 24 Cities')
filename = os.path.join(save_path,"24_cities.png")
plt.savefig(filename)   # save figure to .png in ../results/plots
