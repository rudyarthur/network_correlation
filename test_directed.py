import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from propagate import *
from stats import *
from plots import *

import sys

matplotlib.rcParams['text.usetex'] = True	
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams.update({'font.size': 22})
np.random.seed(1234567)	

fig, ax = plt.subplots(1,3,figsize=(24,8))	

G = nx.DiGraph() 
G.add_edges_from([
(0,1),
(0,2),
(0,3),
(1,3),
(2,3),
(0,4)
])

G.nodes[0]["data"] = 0
G.nodes[1]["data"] = 1
G.nodes[2]["data"] = 1
G.nodes[3]["data"] = 2
G.nodes[4]["data"] = 0



J = np.ones( (3,3) , dtype=int)
for i in range(3):
	for j in range(3):
		J[i,j] = joincount(G, i, j, null=None)

print("J=\n",J)
ax[0].set_title( r'$J = \left(\begin{matrix}' + r'{}&{}&{} \cr {}&{}&{} \cr {}&{}&{}'.format( J[0,0], J[0,1], J[0,2], J[1,0], J[1,1], J[1,2], J[2,0], J[2,1], J[2,2] ) + r'\end{matrix}\right)$', pad=20)			

draw_network_data(G, ax[0], name="data", colorbar=True, draw_labels=True)

###
import random
random.seed(123456789)
G = nx.fast_gnp_random_graph(100, 0.1, seed=123456789, directed=True) #rubbish example, but just need a directed network

propagate(G, 0, source_val=10, start_val = 0, num_steps=10, clean_start=True, name="data", noise=0)
propagate(G, 0, source_val=10, start_val = 0, num_steps=10, clean_start=True, name="data2", noise=0.01)

I, pId, Id  = moran(G, Np=100, null="data", rownorm=True) 
print("I", I, "pdata", pId)
I, pIc, Ic  = moran(G, Np=100, null="config", rownorm=True) 
print("I", I, "pconfig", pIc)
I, pIc, Ic  = moran(G, Np=100, null="config", rownorm=True, name="data2") 
print("I2", I, "pconfig", pIc)

L, pc, dist = lee(G, 'data', 'data2', null="config", Np=100)
print( "Lee", L, "pconfig", pc)




L, pvals, dists = local_moran(G,null="config",Np=100)
Lsig = []
for i in range(len(L)):
	if pvals[i] < 0.01:
		Lsig.append(1)
	else:
		Lsig.append(0)

GL = nx.DiGraph( G )
set_node_data(GL, Lsig)

ax[1].set_title("I = {:.2f}".format(I))
ax[2].set_title("Locally Significant Nodes")
draw_network_data(G, ax[1], name="data", colorbar=False, draw_labels=False)
draw_network_data(GL, ax[2], name="data", colorbar=False, draw_labels=False)

plt.tight_layout()
plt.savefig("directed_stats.png", dpi=fig.dpi,bbox_inches='tight')


