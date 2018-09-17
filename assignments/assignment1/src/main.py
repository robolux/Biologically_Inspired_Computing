# Assignment 1 Main - Biologically Inspired Computing
# 08/29/2018
# Hunter Phillips

import sys
import numpy as nmp
import scipy as scp
import csv


def main():
    data = open_file()
    print(data[1][1])


def open_file():
    with open("european_cities.csv", "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    return data

# def parser(data):

main()
