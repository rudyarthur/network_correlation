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


fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(5, 5,  width_ratios=(1,0.5,0.5,0.5,1), height_ratios=(1,0.5,0.5,0.5,1), left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)

G = nx.karate_club_graph(); 

G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)

source_id = 1
propagate(G, source_id, num_steps=10, noise=0.1)	

ax = fig.add_subplot(gs[1:4, 1:4])
quadrants = moran_scatterplot(G, ax, mean_subtract=False)
ax.set_xlabel("x")
ax.set_ylabel("Ax")	

ax.scatter( quadrants['ul']['x'], quadrants['ul']['y'] , edgecolors='k', color='C0') 
ax.scatter( quadrants['ur']['x'], quadrants['ur']['y'] , edgecolors='k', color='C1') 
ax.scatter( quadrants['ll']['x'], quadrants['ll']['y'] , edgecolors='k', color='C2') 
ax.scatter( quadrants['lr']['x'], quadrants['lr']['y'] , edgecolors='k', color='C3') 



ax = fig.add_subplot(gs[0, 0])
draw_network_data(G, ax, colorbar=False, draw_labels= quadrants['ul']['i'])
ax = fig.add_subplot(gs[0, 4])
draw_network_data(G, ax, colorbar=False, draw_labels= quadrants['ur']['i'])
ax = fig.add_subplot(gs[4, 0])
draw_network_data(G, ax, colorbar=False, draw_labels= quadrants['ll']['i'])
ax = fig.add_subplot(gs[4, 4])
draw_network_data(G, ax, colorbar=False, draw_labels= quadrants['lr']['i'])

		
fig.tight_layout()
plt.savefig("moran_scatter.png", dpi=fig.dpi)
plt.show()









