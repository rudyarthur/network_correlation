import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from propagate import *
from stats import *
from plots import *

import time
import sys

matplotlib.rcParams['text.usetex'] = True	
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams.update({'font.size': 22})
np.random.seed(123456789)	

fig, ax = plt.subplots(1,2,figsize=(16,8))	

G = nx.Graph()
G.add_edge(0,1)
G.add_edge(0,4)
G.add_edge(0,2)
G.add_edge(1,3)
G.add_edge(2,3)
G.nodes[0]["data"] = 0
G.nodes[1]["data"] = 1
G.nodes[2]["data"] = 1
G.nodes[3]["data"] = 2
G.nodes[4]["data"] = 0

J = np.ones( (3,3) , dtype=int)
for i in range(3):
	for j in range(3):
		J[i,j] = joincount(G, i, j, null=None)

print(J)
ax[0].set_title( r'$J = \left(\begin{matrix}' + r'{}&{}&{} \cr {}&{}&{} \cr {}&{}&{}'.format( J[0,0], J[0,1], J[0,2], J[1,0], J[1,1], J[1,2], J[2,0], J[2,1], J[2,2] ) + r'\end{matrix}\right)$', pad=20)			
draw_network_data(G, ax[0], colorbar=True)


G = nx.LFR_benchmark_graph(500, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
#G = nx.karate_club_graph(); 
G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)


propagate(G, 0, source_val=3, num_steps=150, noise=0)	
set_node_data(G, np.rint( get_node_data(G) ).astype(int) ) #make integer data

J = np.ones( (3,3) , dtype=int)
p = np.ones( (3,3) )
for r in range(3):
	for s in range(r,3):
		Jrs, prs, dists = joincount(G, r, s)
		J[r,s] = Jrs
		p[r,s] = prs
		J[s,r] = Jrs
		p[s,r] = prs

print("join counts\n", J)
print("join count pvals\n", p)

draw_network_data(G, ax[1], colorbar=True)
ax[1].set_title( r'$J = \left(\begin{matrix}' + r'{}&{}&{} \cr {}&{}&{} \cr {}&{}&{}'.format( J[0,0], J[0,1], J[0,2], J[1,0], J[1,1], J[1,2], J[2,0], J[2,1], J[2,2] ) + r'\end{matrix}\right)$', pad=20)					

plt.tight_layout()
plt.savefig("joincount.png", dpi=fig.dpi,bbox_inches='tight')


