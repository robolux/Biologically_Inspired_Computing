# Main for Assignment 1 in Biologically Inspired Computing
# Hunter Phillips

import movements
import mlp
import numpy as nmp

filename = '../data/movements_day1-3.dat'
train, train_targets, valid, valid_targets, test, test_targets = data_call(filename=filename)
