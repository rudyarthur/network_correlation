import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stats import get_node_data, set_node_data, copy_node_data, get_adjacency
from scipy.stats import linregress

def draw_network_data(G, ax, name="data", colorbar=False, draw_labels=False):
	nodes = G.nodes
	colors = get_node_data(G, name=name)

	pos = nx.spring_layout(G, seed=123456789)
	ec = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1)
	nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, nodelist=nodes, node_color = colors, vmin = min(0,np.min(colors)), vmax = max(1,np.max(colors)), cmap="rainbow", edgecolors=None)

	if draw_labels: 
		if type(draw_labels) == bool:
			labels=nx.draw_networkx_labels(G,pos, ax=ax)
		elif type(draw_labels) == dict:
			labels=nx.draw_networkx_labels(G,pos, labels=draw_labels,ax=ax,font_weight="bold")
		elif type(draw_labels) ==list:
			labels=nx.draw_networkx_labels(G,pos, labels={i:str(i) for i in draw_labels},ax=ax,font_weight="bold")
			
	if colorbar: plt.colorbar(nc, ax=ax)
	ax.axis('off')

def moran_scatterplot(G, ax, name="data", mean_subtract=False, rownorm=True, drop_weights=False):


	A = get_adjacency(G, rownorm, drop_weights)
	x = get_node_data(G, name)

	if mean_subtract: 
		z = x - np.mean(x)
	else: 
		z = x
	
	zl = A.dot(z) 
	zbar = np.mean(z)
	zlbar = np.mean(zl)


	ax.scatter( z, zl )
	ax.axhline(y=zlbar, color='k', linestyle='--')
	ax.axvline(x=zbar, color='k', linestyle='--')

	result = linregress(z,zl) #slope is moran index	
	xu = np.linspace(min(x), max(x), 101)
	ax.plot(xu, result.slope*xu + result.intercept, color='k', linestyle='dotted')
	
	quadrants = {
	"ll":{"i":[], "x":[], "xl":[]}, 
	"lr":{"i":[], "x":[], "xl":[]}, 
	"ul":{"i":[], "x":[], "xl":[]}, 
	"ur":{"i":[], "x":[], "xl":[]}, 
	}

	for i in range(len(z)):
		if zl[i] > zlbar:
			if z[i] > zbar:
				quadrants['ur']['i'].append(i)
				quadrants['ur']['x'].append(z[i])
				quadrants['ur']['xl'].append(zl[i])
			else:
				quadrants['ul']['i'].append(i)
				quadrants['ul']['x'].append(z[i])
				quadrants['ul']['xl'].append(zl[i])		
		else:
			if z[i] > zbar:
				quadrants['lr']['i'].append(i)
				quadrants['lr']['x'].append(z[i])
				quadrants['lr']['xl'].append(zl[i])			
			else:
				quadrants['ll']['i'].append(i)
				quadrants['ll']['x'].append(z[i])
				quadrants['ll']['xl'].append(zl[i])

				
	return quadrants	


def xogram(G, func, dmin=1, dmax=None, null="data", Np=1000, name="data", smooth=0, rownorm=True, drop_weights=False):
	paths = nx.shortest_path(G)

	if not dmax:
		##figure out max path length
		smax = max( paths.items(), key = lambda s: len(max(s[1].items(), key= lambda d:d[1])[1]) )[0]
		dmax = len( max(paths[smax].items(), key= lambda t:t[1])[1] )

	corr = []
	for d in range(dmin,dmax+1):
		if G.is_directed():
			Gd = nx.DiGraph() 
		else:
			Gd = nx.Graph() 
		Gd.add_nodes_from( G.nodes )

		copy_node_data( G, Gd )
		
		for source, path in paths.items():
			for dest,p in path.items():
				if source == dest: continue

				if len(p) == d+1:
					Gd.add_edge( source, dest )
		

		corr.append( func(Gd, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights) )
	return list(range(dmin,dmax+1)), corr

from stats import moran
def moran_correlogram(G, dmin=1, dmax=None, null="data", Np=1000, name="data",  smooth=0, rownorm=True, drop_weights=True):
	return  xogram(G, moran, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm, drop_weights=drop_weights)

from stats import geary
def geary_correlogram(G, dmin=1, dmax=None, null="data", Np=1000, name="data", smooth=0, rownorm=True, drop_weights=True):
	return  xogram(G, geary, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm, drop_weights=drop_weights)

from stats import getisord
def getisord_correlogram(G, dmin=1, dmax=None, null="data", Np=1000, name="data", smooth=0, rownorm=True, drop_weights=True):
	return  xogram(G, getisord, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm, drop_weights=drop_weights)


def variogram(G, dmin=1, dmax=None, null="data", Np=100, name="data", smooth=0):
	x = get_node_data(G, name=name)
	z = (x - np.mean(x))
	unnorm = (z**2).sum() / (x.shape[0]-1)

	def unnorm_geary(G, name=name, null=null, Np=Np, smooth=smooth, alt="lesser", rownorm=False, return_dists=False, drop_weights=True):
		if null is None:
			return geary(G, name=name, null=null, Np=Np, alt=alt, smooth=smooth, rownorm=False, return_dists=return_dists, drop_weights=True) * unnorm
		
		C, p = geary(G, name=name, null=null, Np=Np, alt=alt, smooth=smooth, rownorm=False, return_dists=return_dists, drop_weights=True) 
		return C*unnorm, p 
		
	return  xogram(G, unnorm_geary, Np=Np, name=name, null=null, smooth=smooth, rownorm=False)

from stats import lee
def lee_correlogram(G, xname, yname, dmin=1, dmax=None, null="data", Np=100, name="data", smooth=0, rownorm=True, drop_weights=False):

	def lee_fix(G, name, null=null, Np=Np, alt="greater", smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights):
		return lee(G, name, yname, null=null, Np=Np, alt=alt, smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights)

	return  xogram(G, lee_fix, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm)

from stats import crossvar
def crossvar_correlogram(G, xname, yname, dmin=1, dmax=None, null="data", Np=100, name="data", smooth=0, rownorm=True, drop_weights=False):

	def crossvar_fix(G, name, null=null, Np=Np, alt="greater", smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights):
		return crossvar(G, name, yname, null=null, Np=Np, alt=alt, smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights)

	return  xogram(G, crossvar_fix, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm)


	
	
	
