import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from community import community_louvain
import matplotlib.patches as mpatches
import sys
import csv
import json
import random

##import stuff from path above
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from propagate import *
from stats import *
from plots import *


matplotlib.rcParams.update({'font.size': 22})

np.random.seed(1234567891)	
random.seed(1234567891)

ego_page = "Network science"
local_significance = 0.01
with open(ego_page + "_network.json", 'r') as infile: links = json.loads( infile.read() )

with open(ego_page + "_pageviews.json", 'r') as infile: pageviews = json.loads( infile.read() )
data = {x['label']:np.log10(x['sum']) for x in pageviews}


with open(ego_page + "_data.json", 'r') as infile: pagedata = json.loads( infile.read() )
watchers = {x:np.log10(v['watchers']) for x,v in pagedata.items()}
page_length = {x:np.log10(v['page_length']) for x,v in pagedata.items()}
total_edits = {x:np.log10(v['total_edits']) for x,v in pagedata.items()}

G = nx.Graph()

nodes = links[ego_page]
for u in nodes:
	for v in links[u]:
		if v in nodes:
			G.add_edge(u,v)

G.remove_edges_from(nx.selfloop_edges(G))
N = len(G.nodes)
print("av short path = ", nx.average_shortest_path_length(G))


alphabetical = sorted( list(G) )
nx.set_node_attributes(G, data, 'data')
nx.set_node_attributes(G, watchers, 'watchers')
nx.set_node_attributes(G, page_length, 'page_length')
nx.set_node_attributes(G, total_edits, 'total_edits')

N = len(G.nodes)
print(N, "nodes")

#pvs = {}
#for n in G.nodes:
#	if n.find("nes") >= 0: print(n)
#	pvs[n] = G.nodes[n]["data"]
#print( sorted(pvs, key=pvs.get) )
#print( list(G.neighbors("Army Research Laboratory")) )		
		

fig, ax = plt.subplots(1,1,figsize=(10,8))
draw_network_data(G, ax, colorbar=True, node_size=20, edge_alpha=0.02, draw_labels=False)


fig.tight_layout()
plt.savefig(ego_page.replace(" ", "") + "pageviews.png", dpi=fig.dpi,bbox_inches='tight')
plt.show()
plt.close()
sys.exit(1)

fig, ax = plt.subplots(1,1,figsize=(10,8))
partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'comm')
draw_network_data(G, ax, colorbar=False, node_size=20, edge_alpha=0.02, draw_labels=False, cmap="rainbow", name="comm")


Ncomm = max(partition.values())

cmap = matplotlib.colormaps.get_cmap('rainbow')
norm = matplotlib.colors.Normalize(vmin=0, vmax=Ncomm)

handles = []
for i in range(Ncomm+1):
	pvs = {}
	for p,c in partition.items():
		if c == i:
			pvs[p] = data[p]
	top = max(pvs, key=pvs.get) 

	if len(pvs) > 20:
		#print( sorted(pvs, key=pvs.get)[-10:] )
		rgba = cmap(norm(i))
		handles.append(  mpatches.Patch(color=rgba, label=top) )
ax.legend(handles=handles, loc="lower left")





fig.tight_layout()
plt.savefig(ego_page.replace(" ", "") + "community.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()
plt.close()


I, pId, Id  = moran(G, Np=200, null="data", nodelist=None)
I, pIc, Ic  = moran(G, Np=200, null="config", nodelist=None) 
print("I = {:.3f}".format(I) , "pd", pId, "pc", pIc) 



fig = plt.figure(figsize=(16, 16))
gs = fig.add_gridspec(5, 5,  width_ratios=(2,1,1,1,2), height_ratios=(2,1,1,1,2), left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)



ax = fig.add_subplot(gs[1:4, 1:4])
quadrants = moran_scatterplot(G, ax, mean_subtract=False)
ax.set_xlabel("x")
ax.set_ylabel("Ax")	


ax.scatter( quadrants['ul']['x'], quadrants['ul']['y'] , edgecolors='k', color='C0') 
ax.scatter( quadrants['ur']['x'], quadrants['ur']['y'] , edgecolors='k', color='C1') 
ax.scatter( quadrants['ll']['x'], quadrants['ll']['y'] , edgecolors='k', color='C2') 
ax.scatter( quadrants['lr']['x'], quadrants['lr']['y'] , edgecolors='k', color='C3') 

pvs = {}
for p in quadrants['ul']['i']: pvs[p] = data[p]
#print( sorted(pvs, key=pvs.get) )

vmin=np.min( get_node_data(G) )
vmax=np.max( get_node_data(G) )

i = 0
for l in ['ul', 'ur', 'll', 'lr']:
	#bigids = np.argsort(quadrants[l]['x'])[:5].astype(int)
	#labs = {quadrants[l]['i'][n]:quadrants[l]['i'][n] for n in bigids}
	ax = fig.add_subplot(gs[ 4*int(i/2), 4*(i%2)])
	draw_network_data(G, ax, colorbar=False, nodelist= quadrants[l]['i'], vmin=vmin, vmax=vmax, node_size=10, edge_alpha=0.02)
	i+=1
	
fig.tight_layout()
plt.savefig(ego_page.replace(" ","") + "pageviewsmoranscatter.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()
plt.close()


L, pvalsd, dists = local_moran(G,null="data",Np=200, nodelist=alphabetical)
L, pvalsc, dists = local_moran(G,null="config",Np=200, nodelist=alphabetical)

for i,n in enumerate( alphabetical ): 
	if pvalsc[i] < 0.01:
		G.nodes[n]["alphac"] = 1
	else:
		G.nodes[n]["alphac"] = 0.1

	if pvalsd[i] < 0.01:
		G.nodes[n]["alphad"] = 1
	else:
		G.nodes[n]["alphad"] = 0.1
		
	G.nodes[n]["L"] = L[i]
	
fig, ax = plt.subplots(1,2,figsize=(16, 8))

draw_network_data(G, ax[0], colorbar=False, node_size=20, edge_alpha=0.02, name="L", cmap="Reds" )
draw_network_data(G, ax[1], colorbar=False, node_size=20, edge_alpha=0.02, name="alphad", cmap="Greens" )
#draw_network_data(G, ax[2], colorbar=False, node_size=20, edge_alpha=0.02, name="alphac", cmap="Greens" )

ax[0].set_title(r"Local Moran $I_i$" )
ax[1].set_title(r"$p_{di}$" )
#ax[2].set_title(r"$p_{ci}$" )

plt.tight_layout()
plt.savefig(ego_page.replace(" ","") + "pageviewslocal.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()
plt.close()




distance, moran_corr, graphs = moran_correlogram(G)
for i,d in enumerate(distance):
	print("I(",d,") =", moran_corr[i], len(graphs[i].edges) )



fig, ax = plt.subplots(3,1,figsize=(8, 24))

dtypes =  ["data", "watchers", "page_length", "total_edits"]
name_dtypes = {"data":"Page views", "watchers":"Watchers", "page_length":"Page Length", "total_edits":"Total Edits"}
for i,name in enumerate(["watchers", "page_length", "total_edits"]):
	I, pId, Id  = moran(G, Np=200, null="data", nodelist=None, name=name)
	draw_network_data(G, ax[i], colorbar=True, cbarsize=1, node_size=20, edge_alpha=0.02, draw_labels=False, name=name)
	ax[i].set_title(name_dtypes[name] + "\n"+ r"$I = {:.3f}$".format(I) + ", " + pformat(pId, 0.01, "p_d") )


fig.tight_layout()
plt.savefig(ego_page.replace(" ", "") + "other.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()
plt.close()


dtypes =  ["data", "watchers", "page_length", "total_edits"]
for i in [0]:
	for j in [1,2,3]:

		rho, p, dist = pearson(G,dtypes[i], dtypes[j])
		L, pdl, dist = lee(G, dtypes[i], dtypes[j], null="data", Np=200)
		L, pcl, dist = lee(G, dtypes[i], dtypes[j], null="config", Np=200)
		

		print(dtypes[i], dtypes[j], "{:.2f}".format(rho), p, end=":")
		sig = ''
		if pdl<0.01: sig = "<=="
		print("{:.2f}".format(L), pdl, pcl, sig)



matplotlib.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(1,1,figsize=(16, 16))
quadrants = moran_scatterplot(G, ax, name="data")

ax.set_xlabel("x", fontsize=22)
ax.set_ylabel("Ax", fontsize=22)	


ax.scatter( quadrants['ul']['x'], quadrants['ul']['y'] , edgecolors='k', color='C0') 
ax.scatter( quadrants['ur']['x'], quadrants['ur']['y'] , edgecolors='k', color='C1') 
ax.scatter( quadrants['ll']['x'], quadrants['ll']['y'] , edgecolors='k', color='C2') 
ax.scatter( quadrants['lr']['x'], quadrants['lr']['y'] , edgecolors='k', color='C3') 
ax.scatter( quadrants['outliers']['x'], quadrants['outliers']['y'] , edgecolors='k', color='k') 
for i in range( len(quadrants['outliers']['x']) ):
	ax.annotate(quadrants['outliers']['i'][i], ( quadrants['outliers']['x'][i], quadrants['outliers']['y'][i]) )


plt.tight_layout()
plt.savefig(ego_page.replace(" ","") + "outliers.png", dpi=fig.dpi,bbox_inches='tight')
#plt.show()
plt.close()
