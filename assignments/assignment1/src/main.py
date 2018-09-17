# Assignment 1 Main - Biologically Inspired Computing
# Hunter Phillips

import sys
import numpy as nmp
import scipy as scp
import csv
from solvers import *

if __name__ == "__main__":

    ic = int(raw_input('How many cities do you want to scan for the optimal path?: '))
    cities, distance = open_file() # open file
    cs, cl = num_cities(ic,distance) # reshape with requested size to compute
    # city size, city list

    # Call Exhaustive
    exhaustive(cities, distance, cs, cl)


    # Call Hill Climb
    nr = int(raw_input('How many runs do you want the hill climber to perform?: '))
    ni = int(raw_input('How many iterations do you want the hill climber to perform?: '))

    hill_climb(cities, distance, cs, cl, nr, ic, ni)
