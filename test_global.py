import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from propagate import *
from stats import *
from plots import draw_network_data
import sys

matplotlib.rcParams.update({'font.size': 22})

np.random.seed(123456789)	

fig, ax = plt.subplots(5,3,figsize=(20,20))	
G1 = nx.LFR_benchmark_graph(100, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
G2 = nx.karate_club_graph(); 
G3 = nx.gnp_random_graph(100, 0.3, seed=np.random, directed=False)

steps = [10,10,30]
sigs = [0.1,0.1,0.1]
for i,G in enumerate([G1, G2, G3]):

	G.remove_edges_from(nx.selfloop_edges(G))
	N = len(G.nodes)
	
	propagate(G, 0, num_steps=steps[i], noise=sigs[i])	

	

	I, pId, Id  = moran(G, Np=100, null="data") 
	C, pCd, Cd  = geary(G, Np=100, null="data") 
	O, pOd, Od  = getisord(G, Np=100, null="data") 
	A, pAd, Ad  = numeric_assortativity(G, Np=100, null="data") 
	
	I, pIc, Ic  = moran(G, Np=100, null="config") 
	C, pCc, Cc  = geary(G, Np=100, null="config") 
	O, pOc, Oc  = getisord(G, Np=100, null="config") 
	A, pAc, Ac  = numeric_assortativity(G, Np=100, null="config") 

	
	draw_network_data(G, ax[0][i])
	ax[0][i].set_title(r"$I = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(I, pId, pIc ) 
	+ "\n" + r"$C = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(C, pCd, pCc ) 
	+ "\n" + r"$G = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(O, pOd, pOc ) 
	+ "\n" + r"$A = {:.2f}$, $p_d = {:.2f}$, $p_c = {:.2f}$".format(A, pAd, pAc ) 
	)

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

	ax[4][i].hist( Ad, bins=np.linspace(min(Ad),max(Ad),31) , alpha=0.25, label=r"$P_d(A)$", density=True)
	ax[4][i].hist( Ac, bins=np.linspace(min(Ac),max(Ac),31) , alpha=0.25, label=r"$P_c(A)$", density=True)
	ax[4][i].set_xlabel("A")
	ax[4][i].set_ylabel("P(A)")
	ax[4][i].legend(loc="upper left")
			
plt.tight_layout()
plt.savefig("global_stats.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()







