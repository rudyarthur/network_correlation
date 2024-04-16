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

fig, ax = plt.subplots(4,3,figsize=(18,18))	
G1 = nx.LFR_benchmark_graph(100, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
G2 = nx.karate_club_graph(); 
G3 = nx.gnp_random_graph(100, 0.3, seed=np.random, directed=False)

steps = [10,10,30]
sigs = [0.1,0.1,0.1]
for i,G in enumerate([G1,G2,G3]):
	print("G",i)
	G.remove_edges_from(nx.selfloop_edges(G))
	N = len(G.nodes)

	#set_random_data(G)
	
	propagate(G, 0, num_steps=steps[i], noise=sigs[i])	


	I, pId, Id  = moran(G, Np=100, null="dist") 
	C, pCd, Cd  = geary(G, Np=100, null="dist") 
	O, pOd, Od  = getisord(G, Np=100, null="dist") 
	
	I, pIc, Ic  = moran(G, Np=100, null="config") 
	C, pCc, Cc  = geary(G, Np=100, null="config") 
	O, pOc, Oc  = getisord(G, Np=100, null="config") 

	draw_network_data(G, ax[0][i])
	ax[0][i].set_title(r"$I = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(I, pId, pIc ) 
	+ "\n" + r"$C = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(C, pCd, pCc ) 
	+ "\n" + r"$G = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(O, pOd, pOc ) )

	ax[1][i].hist( Id, bins=np.linspace(min(Id),max(Id),31) , alpha=0.25, label=r"$P_d(I)$", density=True)
	ax[1][i].hist( Ic, bins=np.linspace(min(Ic),max(Ic),31) , alpha=0.25, label=r"$P_c(I)$", density=True)
	ax[1][i].set_xlabel("I")
	ax[1][i].set_ylabel("P(I)")
	ax[1][i].legend(loc="upper left")

	ax[2][i].hist( Cd, bins=np.linspace(min(Cd),max(Cd),31) , alpha=0.25, label=r"$P_d(C)$", density=True)
	ax[2][i].hist( Cc, bins=np.linspace(min(Cc),max(Cc),31) , alpha=0.25, label=r"$P_c(C)$", density=True)
	ax[2][i].set_xlabel("C")
	ax[2][i].set_ylabel("P(C)")
	ax[2][i].legend(loc="upper left")

	ax[3][i].hist( Od, bins=np.linspace(min(Od),max(Od),31) , alpha=0.25, label=r"$P_d(G)$", density=True)
	ax[3][i].hist( Oc, bins=np.linspace(min(Oc),max(Oc),31) , alpha=0.25, label=r"$P_c(G)$", density=True)
	ax[3][i].set_xlabel("G")
	ax[3][i].set_ylabel("P(G)")
	ax[3][i].legend(loc="upper left")
		
fig.tight_layout()
plt.savefig("global_moran.png", dpi=fig.dpi,bbox_inches='tight')
plt.show()







