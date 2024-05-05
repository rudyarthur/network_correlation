import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stats import get_node_data, set_node_data, copy_node_data, get_adjacency
from scipy.stats import linregress

def draw_network_data(G, ax, seed=123456789, name="data", cbarsize=0.5, vmin=None, vmax=None, nodelist=None, colorbar=False, draw_labels=False, log=False, k=None, draw_edges=True, node_alpha=1, edge_alpha=0.1, node_size=100, cmap='rainbow'):
	
	if nodelist is None:
		nodelist = G.nodes

	colors = get_node_data(G, name=name, nodelist=nodelist)
	if log: colors = np.log10(colors)
	if vmin is None: vmin = np.min(colors)
	if vmax is None: vmax = np.max(colors)
		
	pos = nx.spring_layout(G, seed=seed, k=None)
	if draw_edges:
		ec = nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha)
	nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, nodelist=nodelist, node_color = colors, vmin = vmin, vmax = vmax, cmap=cmap, edgecolors=None, alpha=node_alpha)

	if draw_labels: 
		if type(draw_labels) == bool:
			labels=nx.draw_networkx_labels(G,pos, ax=ax)
		elif type(draw_labels) == dict:
			labels=nx.draw_networkx_labels(G,pos, labels=draw_labels,ax=ax,font_weight="bold")
		elif type(draw_labels) ==list:
			labels=nx.draw_networkx_labels(G,pos, labels={i:str(i) for i in draw_labels},ax=ax,font_weight="bold")
			
	if colorbar: 
		plt.colorbar(nc, ax=ax, shrink=cbarsize)
	ax.axis('off')

def xy_scatterplot(x, y, ax, nodelist, outlier=3):

	xbar = np.mean(x)
	ybar = np.mean(y)

	ax.scatter( x, y )
	ax.axhline(y=ybar, color='k', linestyle='--')
	ax.axvline(x=xbar, color='k', linestyle='--')

	result = linregress(x, y) #slope is moran index	
	xu = np.linspace(min(x), max(x), 101)
	ax.plot(xu, result.slope*xu + result.intercept, color='k', linestyle='dotted', label="{:.3f}".format(result.slope) )
	#ax.legend()
	
	quadrants = {
	"ll":{"i":[], "x":[], "y":[]}, 
	"lr":{"i":[], "x":[], "y":[]}, 
	"ul":{"i":[], "x":[], "y":[]}, 
	"ur":{"i":[], "x":[], "y":[]}, 
	"outliers":{"i":[], "x":[], "y":[]}, 
	}
	

	for i in range(len(x)):
		if y[i] > ybar:
			if x[i] > xbar:
				quadrants['ur']['i'].append(nodelist[i])
				quadrants['ur']['x'].append(x[i])
				quadrants['ur']['y'].append(y[i])
			else:
				quadrants['ul']['i'].append(nodelist[i])
				quadrants['ul']['x'].append(x[i])
				quadrants['ul']['y'].append(y[i])		
		else:
			if x[i] > xbar:
				quadrants['lr']['i'].append(nodelist[i])
				quadrants['lr']['x'].append(x[i])
				quadrants['lr']['y'].append(y[i])			
			else:
				quadrants['ll']['i'].append(nodelist[i])
				quadrants['ll']['x'].append(x[i])
				quadrants['ll']['y'].append(y[i])

	resid = np.abs( (result.slope * x + result.intercept) - y )
	stderr = np.std(resid)
	for i in range(len(x)):
		if resid[i] > outlier*stderr:
			quadrants['outliers']['i'].append(nodelist[i])
			quadrants['outliers']['x'].append(x[i])
			quadrants['outliers']['y'].append(y[i])
				
	return quadrants	

def moran_scatterplot(G, ax, name="data", mean_subtract=False, rownorm=True, drop_weights=False, nodelist=None):
	if nodelist is None:  nodelist = list(G)
		
	A = get_adjacency(G, rownorm=rownorm, drop_weights=drop_weights, nodelist=nodelist)
	x = get_node_data(G, name, nodelist=nodelist)

	if mean_subtract: 
		z = x - np.mean(x)
	else: 
		z = x
	
	zl = A.dot(z) 

	return xy_scatterplot(z, zl, ax, nodelist)

def lee_scatterplot(G, ax, name1, name2, rownorm=True, drop_weights=False, nodelist=None):
	if nodelist is None:  nodelist = list(G)
		
	A = get_adjacency(G, rownorm=rownorm, drop_weights=drop_weights, nodelist=nodelist, loop=1)
	x = get_node_data(G, name1, nodelist=nodelist)
	y = get_node_data(G, name2, nodelist=nodelist)
	
	xl = A.dot(x) 
	yl = A.dot(y) 

	return xy_scatterplot(xl, yl, ax, nodelist)



def xogram(G, func, dmin=1, dmax=None, null="data", Np=1000, name="data", smooth=0, rownorm=True, drop_weights=False):
	paths = nx.shortest_path(G)

	if not dmax:
		##figure out max path length
		smax = max( paths.items(), key = lambda s: len(max(s[1].items(), key= lambda d:d[1])[1]) )[0]
		dmax = len( max(paths[smax].items(), key= lambda t:t[1])[1] )

	corr = []
	graphs = []
	for d in range(dmin,dmax+1):
		if G.is_directed():
			Gd = nx.DiGraph() 
		else:
			Gd = nx.Graph() 
		#Gd.add_nodes_from( G.nodes )
		#copy_node_data( G, Gd )
		
		node_data = {}
		for source, path in paths.items():
			for dest,p in path.items():
				if source == dest: continue

				if len(p) == d+1:
					Gd.add_edge( source, dest )
					node_data[ source ] = G.nodes[ source ][ name ]
					node_data[ dest ] = G.nodes[ dest ][ name ]

		nx.set_node_attributes(Gd, node_data, name)
		graphs.append(Gd)
		corr.append( func(Gd, Np=Np, name=name, null=null, smooth=smooth, rownorm=rownorm, return_dists=False, drop_weights=drop_weights) )
	return list(range(dmin,dmax+1)), corr, graphs

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


	
	
	
