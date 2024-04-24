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


Ix, pdx, Idx = moran(G, name="x", Np=100) 
Iy, pdy, Idy = moran(G, name="y", Np=100) 
Iz, pdz, Idz = moran(G, name="z", Np=100) 

rho_xy, p_xy, dist = pearson(G,"x","y")
rho_xz, p_xz, dist = pearson(G,"x","z")
rho_yz, p_yz, dist = pearson(G,"y","z")

gamma_xy, pg_xy, dist = crossvar(G,"x","y")
gamma_xz, pg_xz, dist = crossvar(G,"x","z")
gamma_yz, pg_yz, dist = crossvar(G,"y","z")



L_xy, pl_xy, dist = lee(G, 'x', 'y', null="data")
L_xz, pl_xz, dist = lee(G, 'x', 'z', null="data")
L_yz, pl_yz, dist = lee(G, 'y', 'z', null="data")

#L_xy, pc_xy, dist = lee(G, 'x', 'y', null="config", Np=10)
#L_xz, pc_xz, dist = lee(G, 'x', 'z', null="config", Np=10)
#L_yz, pc_yz, dist = lee(G, 'y', 'z', null="config", Np=10)		



fig, ax = plt.subplots(1,3,figsize=(18,6))	

draw_network_data(G, ax[0], name="x")
draw_network_data(G, ax[1], name="y")
draw_network_data(G, ax[2], name="z")

ax[0].set_title(
r'$\rho_{xy}$ = '+'{:.2f} pval {:.2f}\n'.format(rho_xy, p_xy)
+r'$\rho_{xz}$ = '+'{:.2f} pval {:.2f}\n'.format(rho_xz, p_xz)
+r'$\rho_{yz}$ = '+'{:.2f} pval {:.2f}\n'.format(rho_yz, p_yz)
+r'$\mathbf{x}$'
)

ax[1].set_title(
r'$\gamma_{xy}$ = '+'{:.2f} pval {:.2f}\n'.format(gamma_xy, pg_xy)
+r'$\gamma_{xz}$ = '+'{:.2f} pval {:.2f}\n'.format(gamma_xz, pg_xz)
+r'$\gamma_{yz}$ = '+'{:.2f} pval {:.2f}\n'.format(gamma_yz, pg_yz)
+r'$\mathbf{y}$'
)

ax[2].set_title(
r'$L{xy}$ = '+'{:.2f} pval {:.2f}\n'.format(L_xy, pl_xy)
+r'$L{xz}$ = '+'{:.2f} pval {:.2f}\n'.format(L_xz, pl_xz)
+r'$L{yz}$ = '+'{:.2f} pval {:.2f}\n'.format(L_yz, pl_yz)
+r'$\mathbf{z}$'
)


#print(r'$\mathbf{u},\mathbf{v}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_xy, p_xy, L_xy, pd_xy, pc_xy))
#print(r'$\mathbf{u},\mathbf{t}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_xz, p_xz, L_xz, pd_xz, pc_xz))
#print(r'$\mathbf{v},\mathbf{t}$ &', '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline'.format(rho_yz, p_yz, L_yz, pd_yz, pc_yz))

fig.tight_layout()
plt.savefig("lee_stats.png", dpi=fig.dpi,bbox_inches='tight')

plt.show()







