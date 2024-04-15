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


fig, ax = plt.subplots(2,4,figsize=(16,8))	
G = nx.karate_club_graph(); 
#G = nx.LFR_benchmark_graph(100, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)

G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)

source_id = 1
propagate(G, source_id, num_steps=10, noise=0.1)	


	
	
#dist null
#Moran
L, pvals, dists = local_moran(G,null="dist",Np=100)
freqs,bins,patches = ax[0][0].hist(L, bins = np.linspace(-0.2,0.2,41), density=True)	
Lbins = []
Lfreqs = []
labs = []
for i in range(N): 
	if pvals[i] < 0.01:
		for b in range(len(freqs)): 
			if L[i] > bins[b] and L[i] < bins[b+1]:
				patches[b].set_fc('C1')
				labs.append(i)

ax[0][0].set_xlabel(r"$I_i$")
ax[0][0].set_ylabel(r"$P(I_i)$")
			
draw_network_data(G, ax[1][0], colorbar=False, draw_labels= labs)

#Geary
L, pvals, dists = local_geary(G,null="dist",Np=100)
freqs,bins,patches = ax[0][2].hist(L, bins = np.linspace(0,0.4,41), density=True)	
labs = []
for i in range(N): 
	if pvals[i] < 0.01:
		for b in range(len(freqs)): 
			if L[i] > bins[b] and L[i] < bins[b+1]:
				patches[b].set_fc('C1')
				labs.append(i)

ax[0][2].set_xlabel(r"$C_i$")
ax[0][2].set_ylabel(r"$P(C_i)$")
			
draw_network_data(G, ax[1][2], colorbar=False, draw_labels= labs)



#config null
#Moran
L, pvals, dists = local_moran(G,null="config",Np=100)
freqs,bins,patches = ax[0][1].hist(L, bins = np.linspace(-0.2,0.2,41), density=True)	
Lbins = []
Lfreqs = []
labs = []
for i in range(N): 
	if pvals[i] < 0.001:
		for b in range(len(freqs)): 
			if L[i] > bins[b] and L[i] < bins[b+1]:
				patches[b].set_fc('C1')
				labs.append(i)

ax[0][1].set_xlabel(r"$I_i$")
ax[0][1].set_ylabel(r"$P_{config}(I_i)$")
				
draw_network_data(G, ax[1][1], colorbar=False, draw_labels= labs)

#Geary
L, pvals, dists = local_geary(G,null="config",Np=100)
freqs,bins,patches = ax[0][3].hist(L, bins = np.linspace(0,0.4,41), density=True)	
Lbins = []
Lfreqs = []
labs = []
for i in range(N): 
	if pvals[i] < 0.001:
		for b in range(len(freqs)): 
			if L[i] > bins[b] and L[i] < bins[b+1]:
				patches[b].set_fc('C1')
				labs.append(i)

ax[0][3].set_xlabel(r"$C_i$")
ax[0][3].set_ylabel(r"$P_{config}(C_i)$")
					
draw_network_data(G, ax[1][3], colorbar=False, draw_labels= labs)


fig.tight_layout()
plt.savefig("local_moran.png", dpi=fig.dpi)
plt.show()













