import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from propagate import *
from stats import *
from plots import *

import time
import sys


matplotlib.rcParams.update({'font.size': 22})
np.random.seed(123456789)	

fig, ax = plt.subplots(1,2,figsize=(16,8))	


G = nx.LFR_benchmark_graph(500, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
#G = nx.karate_club_graph(); 
G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)




propagate(G, 0, source_val=3, num_steps=150, noise=0)	
propagate(G, 0, source_val=3, num_steps=150, noise=0, name="data2")	
set_node_data(G, np.rint( get_node_data(G) ).astype(int) )
set_node_data(G, np.rint( get_node_data(G, name="data2") ).astype(int), name="data2" )
set_random_data(G, name="data3")

C, p, dists = crossvar(G, "data", "data2")
print("crossvar_12 = ", C, "pval", p)
C, p, dists = crossvar(G, "data", "data3")
print("crossvar_13 = ", C, "pval", p)

J = np.ones( (3,3) )
p = np.ones( (3,3) )
pC = np.ones( (3,3) )
for r in range(3):
	for s in range(r,3):
		Jrs, prs, dists = joincount(G, r, s)
		J[r,s] = Jrs
		p[r,s] = prs
		J[s,r] = Jrs
		p[s,r] = prs

print("join counts\n", J)
print("join count pvals\n", p)
		
draw_network_data(G, ax[0], colorbar=True)
draw_network_data(G, ax[1], colorbar=True, name="data2")
plt.show()


