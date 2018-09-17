# Assignment 1 Read File - Biologically Inspired Computing
# Hunter Phillips

import sys
import numpy as nmp
import csv

# open csv file
def open_file():

    data = open('european_cities.csv', 'r+') # open file and assign to data
    reader = csv.reader(data, delimiter=';') # set delimiter

    # init arrays
    cities = []
    distance = nmp.zeros((24, 24))

    # set counter
    m = 0

    # assign arrays
    for val in reader:
        if m == 0:
            for d in range(len(val)):
                cities.append(val[d])
        else:
            for d in range(len(distance)):
                distance[m - 1][d] = float(val[d])
        m = m + 1

    return cities, distance

    
