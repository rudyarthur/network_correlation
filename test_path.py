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

fig, ax = plt.subplots(2,1,figsize=(8,16))	

G = nx.LFR_benchmark_graph(500, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
#G = nx.karate_club_graph(); 
#G = nx.gnp_random_graph(100, 0.3, seed=np.random, directed=False)

G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)

propagate(G, 0, num_steps=50, noise=0.1)	

##correlogram
distance, moran_corr = moran_correlogram(G, moran)

sig=0.01
ax[0].plot(distance, [m[0] for m in moran_corr], c="C0", ls='--')
ax[0].scatter([d for i,d in enumerate(distance) if moran_corr[i][1] <= sig], [m[0] for m in moran_corr if m[1] <= sig], c="C0", s=120)
ax[0].scatter([d for i,d in enumerate(distance) if moran_corr[i][1] > sig], [m[0] for m in moran_corr if m[1] > sig], s=120, edgecolor="C0", facecolor="None")

	

draw_network_data(G, ax[1], colorbar=False)

plt.show()
