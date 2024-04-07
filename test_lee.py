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

G = nx.LFR_benchmark_graph(100, 2.1, 1.5, 0.05, average_degree=10, max_degree=30, min_community=20, seed=123456789)
G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)

#set_random_data(G)
source_x = 0
source_y = 44
source_z = 43
propagate(G, source_x, num_steps=10, name="x", noise=0.1)	
propagate(G, source_y, num_steps=10, name="y", noise=0.1)	
propagate(G, source_x, num_steps=5, name="z", noise=None)	
propagate(G, source_y, num_steps=5, name="z", clean_start=False, noise=0.1)	


Ix, pdx, Idx = moran(G, name="x", Np=10) 
Iy, pdy, Idy = moran(G, name="y", Np=10) 
Iz, pdz, Idz = moran(G, name="z", Np=10) 


rho_xy, p_xy, dist = pearson(G,"x","y")
rho_xz, p_xz, dist = pearson(G,"x","z")
rho_yz, p_yz, dist = pearson(G,"y","z")



L_xy, pd_xy, dist = lee(G, 'x', 'y', null="data")
L_xy, pc_xy, dist = lee(G, 'x', 'y', null="config", Np=10)

L_xz, pd_xz, dist = lee(G, 'x', 'z', null="data")
L_xz, pc_xz, dist = lee(G, 'x', 'z', null="config", Np=10)

L_yz, pd_yz, dist = lee(G, 'y', 'z', null="data")
L_yz, pc_yz, dist = lee(G, 'y', 'z', null="config", Np=10)		



fig, ax = plt.subplots(1,3,figsize=(18,6))	

draw_network_data(G, ax[0], name="x")
draw_network_data(G, ax[1], name="y")
draw_network_data(G, ax[2], name="z")
ax[0].set_title(r"$\mathbf{u}$"+": $I_u = {:.2f}$".format(Ix))
ax[1].set_title(r"$\mathbf{v}$"+": $I_v = {:.2f}$".format(Iy))
ax[2].set_title(r"$\mathbf{t}$"+": $I_t = {:.2f}$".format(Iz))



print(r'$\mathbf{u},\mathbf{v}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_xy, p_xy, L_xy, pd_xy, pc_xy))
print(r'$\mathbf{u},\mathbf{t}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_xz, p_xz, L_xz, pd_xz, pc_xz))
print(r'$\mathbf{v},\mathbf{t}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_yz, p_yz, L_yz, pd_yz, pc_yz))

fig.tight_layout()
plt.savefig("lee_statistic.png", dpi=fig.dpi,bbox_inches='tight')

plt.show()







