# Assignment 1 Shaping Arrays to Size - Biologically Inspired Computing
# Hunter Phillips

import sys
import numpy as nmp
import csv

def city_names(des_length, cities): # desired length and how many cities to place

    name_list = [] # init blank var

    for m in range(len(des_length)):
        name_list.append(cities[int(des_length[m])])

    return name_list

def num_cities(s, d):

    reshape = d[0:s, 0:s]
    reshape_new = nmp.zeros(len(reshape))
    for m in range(len(reshape)):
        reshape_new[m] = m
    return reshape, reshape_new

def trip_length(path, small_city):

    alpha_trip = nmp.zeros(len(small_city))

    for m in range(len(small_city) - 1):
        alpha_trip[m] = small_city[int(path[m]), int(path[m + 1])]
    alpha_trip[-1] = small_city[int(path[-1]), int(path[0])]
    beta_trip = nmp.sum(alpha_trip)
    return beta_trip
