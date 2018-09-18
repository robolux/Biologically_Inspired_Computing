# Assignment 1 Testing - Biologically Inspired Computing
# Hunter Phillips

import sys
import os.path
from solvers import *
import numpy as nmp
import time

# Exhaustive Algorithm Text Output

cities, distance = open_file()
range1 = nmp.linspace(6,10,5)
range1 = [str(int(i)) for i in range1]

save_path = '../results/exhaustive'

for w, g in enumerate(range1, 1):
    filename = os.path.join(save_path, "exhaustive_" + str(g) + ".txt")
    print(filename)
    with open (filename, 'w') as f_m:           # with auto closes files which is helpful here

        cs, cl = num_cities(int(g),distance)
        nc, ideal_trip = exhaustive(cities, distance, cs, cl)
        city_header = ', '.join(str(h) for h in nc)
        f_m.write(city_header)
        f_m.write('\n\n')
        f_m.write(str(ideal_trip))


# Hill Climbing Algorithm Text Output

cities, distance = open_file()
range2 = nmp.linspace(6,24,19)
range2 = [str(int(i)) for i in range2]

save_path = '../results/hillclimb'

for w2, g2 in enumerate(range2, 1):

    filename = os.path.join(save_path, "hill_" + str(g2) + ".txt")
    print(filename)
    with open (filename, 'w') as f_m:

        ic = int(g2)
        cs, cl = num_cities(ic,distance)
        bt_array = hill_climb(cities, distance, cs, cl, 20, ic, 20)
        f_m.write('Best Trip\n\n')
        bt_val = ', \n'.join(str(h) for h in bt_array)
        f_m.write(bt_val)

        f_m.write('\n\n\nBest Trip of All Runs\n\n')
        f_m.write(str(nmp.min(bt_array)))

        f_m.write('\n\n\nWorst Trip of All Runs\n\n')
        f_m.write(str(nmp.max(bt_array)))

        average_runs_besttrip = nmp.mean(bt_array, axis = 0)
        f_m.write('\n\n\nAverage Runs of Best Trip\n\n')
        f_m.write(str(average_runs_besttrip[0]))

        sd = nmp.std(bt_array) # Calculate Standard Deviation
        f_m.write('\n\n\nStandard Deviation Across Runs\n\n')
        f_m.write(str(sd))


# Genetic Algorithm Text Output

cities, distance = open_file()
range3   = nmp.linspace(6,24,19)
range3   = [str(int(i)) for i in range3]
pop_size = ['10', '40', '160']

save_path = '../results/ga'

for w3, g3 in enumerate(range3, 1):
    for ee, popu in enumerate(pop_size, 1):
        filename = os.path.join(save_path, "ga_" + str(g3) + "_pop_" + popu + ".txt")
        print(filename)
        with open (filename, 'w') as f_m:

            ic = int(g3)
            ps = int(popu)
            cs, cl = num_cities(ic, distance)
            var_hold, nr_bt = GA(cities, distance, cs, cl, ps, 100, 0.2, 20, 10, ic)

            av_ga = nmp.mean(var_hold, axis = 0)
            f_m.write('Best Fit Average for Each Generation\n\n')
            av_out = ', \n'.join(str(h) for h in av_ga)
            f_m.write(str(av_ga))

            f_m.write('\n\n\nBest Trip\n\n')
            nr_val = ', \n'.join(str(h) for h in nr_bt)
            f_m.write(nr_val)

            f_m.write('\n\n\nBest Trip of All Runs\n\n')
            f_m.write(str(nmp.min(nr_bt)))

            f_m.write('\n\n\nWorst Trip of All Runs\n\n')
            f_m.write(str(nmp.max(nr_bt)))

            average_runs_besttrip2 = nmp.mean(nr_bt, axis = 0)
            f_m.write('\n\n\nAverage Runs of Best Trip\n\n')
            f_m.write(str(average_runs_besttrip2[0]))

            sd2 = nmp.std(nr_bt) # Calculate Standard Deviation
            f_m.write('\n\n\nStandard Deviation Across Runs\n\n')
            f_m.write(str(sd2))



####### Benchmarking ######


# Exhaustive
save_path = '../results/bench'
filename = os.path.join(save_path, "exhaustive_bench.txt")
print(filename)
f_n = open(filename,'w')

for w4, g4 in enumerate(range1, 1):
    cs, cl = num_cities(int(g4),distance)

    start_time = time.time()
    nc, ideal_trip = exhaustive(cities, distance, cs, cl)
    end_time = time.time()

    general_time = end_time - start_time

    f_n.write('Number of Cities: ' + str(g4) + '\nTotal Time: ' + str(general_time) + ' seconds')
    f_n.write('\n\n\n')

f_n.close()

# Hill Climbing
save_path = '../results/bench'
filename = os.path.join(save_path, "hillclimb_bench.txt")
print(filename)
f_o = open(filename,'w')

for w5, g5 in enumerate(range2, 1):
    cs, cl = num_cities(int(g5),distance)
    ic = int(g5)

    start_time = time.time()
    bt_array = hill_climb(cities, distance, cs, cl, 20, ic, 20)
    end_time = time.time()

    general_time = end_time - start_time

    f_o.write('Number of Cities: ' + str(g5) + '\nTotal Time: ' + str(general_time) + ' seconds')
    f_o.write('\n\n\n')

f_o.close()


# Genetic Algorithm
save_path = '../results/bench'
filename = os.path.join(save_path, "ga_bench.txt")
print(filename)
f_p = open(filename,'w')
trsh = 10

for ee, popu in enumerate(pop_size, 1):

    f_p.write('Population = ' + str(int(popu)) + '\n*************************************\n\n')

    for w6, g6 in enumerate(range3, 1):

        cs, cl = num_cities(int(g6),distance)
        ic = int(g6)
        ps = int(popu)

        start_time = time.time()
        var_hold, nr_bt = GA(cities, distance, cs, cl, ps, 100, 0.2, 20, 10, ic)
        end_time = time.time()

        general_time = end_time - start_time

        f_p.write('Number of Cities: ' + str(g6) + '\nTotal Time: ' + str(general_time) + ' seconds')
        f_p.write('\n\n\n')

f_p.close()
