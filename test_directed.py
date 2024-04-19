import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from propagate import *
from stats import *
from plots import *

import sys

matplotlib.rcParams.update({'font.size': 22})

np.random.seed(123456789)	

fig, ax = plt.subplots(1,3,figsize=(24,8))	
G = nx.fast_gnp_random_graph(100, 0.1, seed=123456789, directed=True) #rubbish example, but just need a directed network
#set_random_data(G)

#G = nx.DiGraph() 
#G.add_edges_from([
#(0,1),
#(0,2),
#(0,3),
#(1,3),
#(2,3),
#])

#propagate(G, 0, source_val=1, start_val = 0, num_steps=1, clean_start=True, name="data", noise=0)
propagate(G, 0, source_val=10, start_val = 0, num_steps=10, clean_start=True, name="data", noise=0)
propagate(G, 0, source_val=10, start_val = 0, num_steps=9, clean_start=True, name="data2", noise=0)

I, pId, Id  = moran(G, Np=100, null="data", rownorm=True) 
print("G", I, pId)
I, pId, Id  = moran(G, Np=100, null="data", rownorm=True, name="data2")  
print("H", I, pId)

Lee, pc, dist = lee(G, 'data', 'data2', null="config", Np=100)
print( "Lee", Lee, pc)

I, pIc, Ic  = moran(G, Np=100, null="config", rownorm=False) 
print(I, pIc)
L, pvals, dists = local_moran(G,null="config",Np=100)
print(L, pvals)
GL = nx.DiGraph( G )
set_node_data(GL, L)



##drawing works by default (arrows added if directed)
draw_network_data(G, ax[0], name="data", colorbar=False, draw_labels=True)
draw_network_data(G, ax[1], name="data2", colorbar=False, draw_labels=True)
draw_network_data(GL, ax[2], name="data", colorbar=False, draw_labels=True)
plt.show()

