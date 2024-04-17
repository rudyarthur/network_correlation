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

##moran correlogram, but can swap with geary or getisord
dist, moran_corr = correlogram(G, moran)
ax[0].bar(dist, moran_corr)

draw_network_data(G, ax[1], colorbar=False)

plt.show()