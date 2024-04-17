import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stats import get_node_data, set_node_data, get_adjacency
from scipy.stats import linregress

def draw_network_data(G, ax, name="data", colorbar=False, draw_labels=False):
	nodes = G.nodes
	colors = get_node_data(G, name=name)

	pos = nx.spring_layout(G, seed=123456789)
	ec = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1)
	nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, nodelist=nodes, node_color = colors, vmin = min(0,np.min(colors)), vmax = 1, cmap="rainbow", edgecolors=None)

	if draw_labels: 
		if type(draw_labels) == bool:
			labels=nx.draw_networkx_labels(G,pos, ax=ax)
		elif type(draw_labels) == dict:
			labels=nx.draw_networkx_labels(G,pos, labels=draw_labels,ax=ax,font_weight="bold")
		elif type(draw_labels) ==list:
			labels=nx.draw_networkx_labels(G,pos, labels={i:str(i) for i in draw_labels},ax=ax,font_weight="bold")
			
	if colorbar: plt.colorbar(nc, ax=ax)
	ax.axis('off')
	
def moran_scatterplot(G, ax, mean_subtract=False):


	A = get_adjacency(G)
	x = get_node_data(G)

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
	
def correlogram(G, func, dmin=1, dmax=None):
	paths = nx.shortest_path(G)

	if not dmax:
		##figure out max path length
		smax = max( paths.items(), key = lambda s: len(max(s[1].items(), key= lambda d:d[1])[1]) )[0]
		dmax = len( max(paths[smax].items(), key= lambda t:t[1])[1] )

	corr = []
	for d in range(dmin,dmax+1):
		Gd = nx.Graph() 
		Gd.add_nodes_from( G.nodes )
		set_node_data( Gd, get_node_data(G) )
	
		for source, path in paths.items():
			for dest,p in path.items():
				if source == dest: continue

				if len(p) == d+1:
					Gd.add_edge( source, dest )
		

		corr.append( func(Gd, null=None) )
	return list(range(dmin,dmax+1)), corr





	
	
	
