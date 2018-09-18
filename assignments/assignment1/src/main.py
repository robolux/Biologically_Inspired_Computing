# Assignment 1 Main - Biologically Inspired Computing
# Hunter Phillips

import sys
import numpy as nmp
import scipy as scp
import csv
from solvers import *

if __name__ == "__main__":

    print('\nWelcome to the Traveling Salesman Problem Algorithmic Solver Set Interface')
    print('***************************************************************************')
    print('The program will compute the shortest distance between a given number of')
    print('cities given in the provided csv file. This computation will be performed')
    print('by exhaustive brute force, hill climbing, and genetic algorithms.\n')

    ic = int(raw_input('\nHow many cities do you want to scan for the optimal path?: '))
    cities, distance = open_file() # open file
    cs, cl = num_cities(ic,distance) # reshape with requested size to compute
    # city size, city list

    # Call Exhaustive
    nc, ideal_trip = exhaustive(cities, distance, cs, cl)
    print('\n')

    # Call Hill Climb
    nr = int(raw_input('How many runs do you want the hill climber to perform?: '))
    ni = int(raw_input('How many iterations do you want the hill climber to perform?: '))
    print('\n')

    list_bt = hill_climb(cities, distance, cs, cl, nr, ic, ni)

    # Call GA
    ps  = int(raw_input('What population size do you want the GA to have?: '))
    ts  = int(raw_input('What tournament size do you want the GA to have?: '))
    ng  = int(raw_input('How many generations do you want the GA to have?: '))
    prb = float(raw_input('What probability do you want the GA to have?: '))
    nr2 = int(raw_input('How many runs do you want the GA to perform?: '))

    var_hold, nr_bt = GA(cities, distance, cs, cl, ps, ng, prb, nr2, ts, ic)

    # give user pretty outputs
    print("\n\n\nEXHAUSTIVE RESULTS\n******************************\n")
    print("Best Trip: " + str(ideal_trip) + '\n\n\n')

    print("\nHILL CLIMBING RESULTS\n******************************\n")
    print("Best Trip: " + str(nmp.min(list_bt)) + '\n\n')
    print("Worst Trip: " + str(nmp.max(list_bt)) + '\n\n')
    avg_bt_1 = nmp.mean(list_bt, axis = 0)
    print("Average Trip: " + str(avg_bt_1[0]) + '\n\n')
    print("Standard Deviation: " + str(nmp.std(list_bt)) + '\n\n\n')

    print("\nGENETIC ALGORITHM RESULTS\n******************************\n")
    print("Best Trip: " + str(nmp.min(nr_bt)) + '\n\n')
    print("Worst Trip: " + str(nmp.max(nr_bt)) + '\n\n')
    avg_bt_2 = nmp.mean(nr_bt, axis = 0)
    print("Average Trip: " + str(avg_bt_2[0]) + '\n\n')
    print("Standard Deviation: " + str(nmp.std(nr_bt)) + '\n\n\n')
