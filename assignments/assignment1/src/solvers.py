# Assignment 1 Solvers - Biologically Inspired Computing
# Hunter Phillips

from read import *
from shape import *
import random
import numpy as nmp
import scipy as scp
from itertools import permutations

# Exhaustive Search Function
def exhaustive(cities, distance, cs, cl):

    ideal_trip = 1 # init shortest distance for the TSP
    l_cs = len(cs)

    for m in range(0, l_cs-1):
        ideal_trip = ideal_trip + cs[m][m+1]
        # print(ideal_trip)
    ideal_trip = ideal_trip + cs[0][l_cs-1]

    cl_new = cl
    for m in permutations(cl_new):
        gamma_trip = trip_length(m, cs)
        if gamma_trip < ideal_trip:
            ideal_trip = gamma_trip
            cl_new = m

    nc = city_names(cl_new, cities)

    return nc, ideal_trip


# Hill Climbing Function
def hill_climb(cities, distance, cs, cl, nr, ic, ni):

    size = ic

    def climb(rc, ni):

        bt = trip_length(rc, cs)
        m  = rc.copy()
        count = 0
        while count < ni:
            counter_2 = 0
            while counter_2 == 0:
                a = nmp.random.randint(len(rc))
                b = nmp.random.randint(len(rc))
                if a != b:
                    counter_2 = 1
            c1 = rc[a]
            c2 = rc[b]
            rc[b] = c1
            rc[a] = c2
            delta_trip = trip_length(rc, cs)
            if delta_trip < bt:
                m  = rc.copy()
                bt = delta_trip
            else:
                rc = m.copy()
            count += 1
        return bt, m

    rc = cl
    nmp.random.shuffle(rc)

    list_bt = nmp.zeros((nr, 1))            # list best trip
    list_bt2 = nmp.zeros((nr, size))        # list best tour

    for run in range(0, nr):

        bt, best_tour = climb(rc, ni)
        # print(best_tour)
        list_bt[run] = bt
        for i in range(size):
            list_bt2[run][i] = best_tour[i]

    return list_bt

# Genetic Algorithm Function
def GA(cities, distance, cs, cl, ps, ng, prb, nr2, ts, ic):

    ts   = ts   # There is some redundency here, but for some odd reason
    nr2  = nr2  # the solutions tend to be worse when I remove it
    prb  = prb  # it may be just my imagination, but if I have learned one
    ps   = ps   # thing whilst programming, magic exists.
    size = ic
    ng   = ng
    cl   = cl
    cs   = cs

    def mk_p(size,cl):
    	p = nmp.zeros((size, len(cl)))
    	for i in range(0,size):
    		rc   = cl
    		nmp.random.shuffle(rc)
    		p[i] = rc

    	return p


    def mk_t(ts,p):

    	tl   = nmp.zeros(( p.shape[0]/ts,ts))
    	temp = nmp.arange(p.shape[0], dtype=nmp.int)

    	nmp.random.shuffle(temp)
    	pol = 0
    	for i in range(0,int(p.shape[0]/ts)):
    		for j in range(0,int(ts)):
    			tl[i][j] = temp[pol]
    			pol +=1


    	parents = nmp.zeros((tl.shape[0],2))

    	trash0 = nmp.zeros(tl.shape[0])
    	for i in range(0,tl.shape[0]):
    		temp_parentlist = nmp.zeros(tl.shape[1])

    		for j in range(0,tl.shape[1]):

    			xxx  = p[int(tl[i][j])]       # I am getting tired of variable names
    			xxxx = [trip_length(xxx,cs)] # at this point
    			nmp.put(temp_parentlist, [j], xxxx)

    		k = nmp.argpartition(temp_parentlist,-2)[:2]
    		trash0[i] = nmp.argpartition(temp_parentlist,-2)[-1:]
    		parents[i][0] = tl[i][k[0]]
    		parents[i][1] = tl[i][k[1]]

    	trash1 = nmp.zeros(trash0.shape[0])

    	for i in range(len(trash0)):
    		trash1[i] = tl[i][int(trash0[i])]

    	trash1 = sorted(trash1, reverse = False)

    	return parents,trash1

    def mk_c(p,parents):

    	pol = 0
    	c1  = nmp.zeros((parents.shape[0], p.shape[1]))

    	for i in range(parents.shape[0]):

    		p1 = p[int(parents[i][0])]

    		p2 = p[int(parents[i][1])]

    		sizep1 = len(p1)

    		r2 = nmp.random.randint(sizep1) + 1
    		p11 = sizep1-r2 + 1

    		p22 = nmp.random.randint(p11)

    		seg = []
    		for i in range(p22,p22+r2):
    			seg.append(p1[i])
    		j=0
    		c2 = p2
    		for i in range(p22,p22+r2):

    			if seg[j] in p2:
    				mt = nmp.nonzero(p2 == seg[j])
    				p2[mt] = p2[i]

    			c2[i] = seg[j]

    			j += 1

    		for m in range(len(c2)):
    			c1[pol][m] = c2[m]
    		pol +=1
    	return c1

    def mk_m(c1,prb):

    	for i in range(c1.shape[0]):
    		if random.random() <= prb:

    			mutate_size = nmp.random.randint(c1.shape[1])+1
    			pointe = nmp.random.randint(c1.shape[1]-mutate_size+1)
    			endpointe = pointe + mutate_size
    			segment = c1[i][pointe:endpointe]
    			reve_mut = segment[::-1]
    			c1[i][pointe:endpointe] = reve_mut

    	return c1

    def mk_newp(p, c1, trash1):

    	newp = nmp.zeros(p.shape)
    	pol  = 0
    	for i in range(newp.shape[0]):
    		if i not in trash1:
    			for j in range(newp.shape[1]):
    				newp[i][j] = p[i][j]

    		if float(i) in trash1:
    			for m in range(p.shape[1]):
    				newp[i][m] = c1[pol][m]
    			pol +=1


    	return newp

    var_hold = nmp.zeros((nr2,ng))

    nr_bt = nmp.zeros((nr2, 1))

    for run in range(0,nr2):
    	br = nmp.zeros((ng, size)) 							#Best Route in Each Generation
    	bl = nmp.zeros(ng) 									#Best Length in Each Generation
    	p  = mk_p(ps,cl)

    	for i in range(ng):

    		parents,trash1 = mk_t(ts,p)
    		c1=  mk_c(p,parents)
    		c1 =  mk_m(c1,prb)

    		p = mk_newp(p, c1, trash1)
    		ideal_trip = trip_length(p[0],cs)
    		m = p[0]

    		for j in range(p.shape[0]):
    			delta_trip1 = trip_length(p[j],cs)
    			if delta_trip1 < ideal_trip:
    				ideal_trip = delta_trip1
    				m = p[j]

    		for k in range(p.shape[1]):
    			br[i][k] = m[k]

    			bl[i] = ideal_trip

    	for i in range(ng):

    		var_hold[run][i] = bl[i]

    	ideal_trip = trip_length(p[0],cs)
    	for i in range(p.shape[0]):
    		delta_trip1 = trip_length(p[i],cs)
    		if delta_trip1 < ideal_trip:
    			ideal_trip = delta_trip1
    			m = p[i]

    	nr_bt[run] = ideal_trip

    return var_hold, nr_bt
