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

fig, ax = plt.subplots(2,3,figsize=(18,12))	
G1 = nx.LFR_benchmark_graph(100, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
G2 = nx.karate_club_graph(); 
G3 = nx.gnp_random_graph(100, 0.3, seed=np.random, directed=False)

steps = [10,10,30]
sigs = [0.1,0.1,0.1]
for i,G in enumerate([G1,G2,G3]):
	G.remove_edges_from(nx.selfloop_edges(G))
	N = len(G.nodes)

	#set_random_data(G)
	
	propagate(G, 0, num_steps=steps[i], noise=sigs[i])	
	#x = get_node_data(G) + np.random.normal(0,sigs[i],size=N)
	#set_node_data(G,x)

	I, pd, Id  = moran(G, Np=100, null="data") 
	I, pc, Ic  = moran(G, Np=100, null="config") 
	
	draw_network_data(G, ax[0][i])
	ax[0][i].set_title(r"$I = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(I, pd, pc ))

	ax[1][i].hist( Id, bins=np.linspace(min(Id),max(Id),31) , alpha=0.25, label=r"$P_d(I)$", density=True)
	ax[1][i].hist( Ic, bins=np.linspace(min(Ic),max(Ic),31) , alpha=0.25, label=r"$P_c(I)$", density=True)
	ax[1][i].set_xlabel("I")
	ax[1][i].set_ylabel("P(I)")
	ax[1][i].legend(loc="upper left")

fig.tight_layout()
plt.savefig("global_moran.png", dpi=fig.dpi,bbox_inches='tight')
plt.show()







