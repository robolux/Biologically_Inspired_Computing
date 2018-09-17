# Assignment 1 Solvers - Biologically Inspired Computing
# 08/29/2018
# Hunter Phillips

from read import *
from shape import *
import numpy as nmp
import scipy as scp
from itertools import permutations

# Exhaustive Search Function
def exhaustive(cities, distance, cs, cl):

    ideal_trip = 1 # init shortest distance for the TSP
    l_cs = len(cs)

    for m in range(0, l_cs-1):
        ideal_trip = ideal_trip + cs[m][m+1]
    ideal_trip = ideal_trip + cs[0][l_cs-1]

    cl_new = cl
    for m in permutations(cl_new):
        gamma_trip = trip_length(m, cs)
        if gamma_trip < ideal_trip:
            ideal_trip = gamma_trip
            cl_new = m

    citynames = city_names(cl_new, cities)

    print(citynames)
    print(ideal_trip)


# Hill Climbing Function
def hill_climb(cities, distance, cs, cl, nr, ic, ni):

    rc = cl                 # randomize cities with new var
    print(rc)
    nmp.random.shuffle(rc)  # perform operation
    print(rc)

    bt_array = nmp.zeros((nr,1))    # best trip
    br_array = nmp.zeros((nr,ic))   # best run

    for r in range(0,nr):

        bt = trip_length(rc, cs)

        m = rc.copy()
        count = 0

        while count < ni:
            count2 = 0
            while count2 ==0:
                a = nmp.random.randint(len(rc))
                b = nmp.random.randint(len(rc))
                if a!=b:
                    count2=1
            c1    = rc[a]
            c2    = rc[b]
            rc[b] = c1
            rc[a] = c2
            testtrip = trip_length(rc,cs)
            if testtrip < bt:
                m  = rc.copy()
                bt = testtrip
            else:
                rc = m.copy()
            count +=1

        bt_array[r] = bt
        for tt in range(ic):
            br_array[r][tt] = m[tt]
            
    print(bt_array)

# Genetic Algorithm Function
# def GA():
