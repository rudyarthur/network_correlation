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
#G = nx.gnp_random_graph(100, 0.3, seed=np.random, directed=False)

G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)

print("av short path = ", nx.average_shortest_path_length(G))

propagate(G, 0, source_val=1, num_steps=250, noise=0.1, name="data")	
#set_node_data(G, np.rint( get_node_data(G) ).astype(int) ) #make integer data
#propagate(G, 0, source_val=3, num_steps=150, noise=0, name="data2")	
#set_node_data(G, np.rint( get_node_data(G) ).astype(int), name="data2" ) #make integer data

sig=0.01


##moran correlogram
distance, moran_corr, graphs = moran_correlogram(G)

#ax[1].set_title("Moran Correlogram")
ax[1].set_ylabel("I(d)")
ax[1].set_xlabel("d")

ax[1].plot(distance, [m[0] for m in moran_corr], c="C0", ls='--')
ax[1].scatter([d for i,d in enumerate(distance) if moran_corr[i][1] <= sig], [m[0] for m in moran_corr if m[1] <= sig], c="C0", s=160)
ax[1].scatter([d for i,d in enumerate(distance) if moran_corr[i][1] > sig], [m[0] for m in moran_corr if m[1] > sig], s=160, edgecolor="C0", facecolor="None")


#ax[0].set_title(r"$\mathbf{u}$")
draw_network_data(G, ax[0], colorbar=True, cbarsize=1)

"""
distance, lee_corr = lee_correlogram(G, "data", "data2")

ax[1][1].set_title("Lee Correlogram")

ax[1][1].plot(distance, [m[0] for m in lee_corr], c="C0", ls='--')
ax[1][1].scatter([d for i,d in enumerate(distance) if lee_corr[i][1] <= sig], [m[0] for m in lee_corr if m[1] <= sig], c="C0", s=120)
ax[1][1].scatter([d for i,d in enumerate(distance) if lee_corr[i][1] > sig], [m[0] for m in lee_corr if m[1] > sig], s=120, edgecolor="C0", facecolor="None")


ax[0][1].set_title(r"$\mathbf{v}$")	
draw_network_data(G, ax[0][1], colorbar=True, name="data2")
"""

plt.tight_layout()
plt.savefig("correlogram.png", dpi=fig.dpi,bbox_inches='tight')

plt.show()
