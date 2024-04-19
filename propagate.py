import networkx as nx
import numpy as np
from stats import get_node_data, set_node_data

def set_random_data(G, name="data", randomfn=None):
	if randomfn:
		x = randomfn(len(G.nodes))
	else:
		x = np.random.random(len(G.nodes))
	nx.set_node_attributes(G, {n:x[i] for i,n in enumerate(G.nodes)}, name)

def propagate(G, source_id, source_val=1, start_val = 0, num_steps=10, clean_start=True, name="data", noise=0.1):
	if clean_start:
		for n in G.nodes: G.nodes[n][name] = start_val
		G.nodes[source_id][name] = source_val	
	for i in range(num_steps):
		if G.is_directed():
			nv = { n:np.mean( [ G.nodes[k][name] for k in G.predecessors(n) ] ) for n in G.nodes if len( list(G.predecessors(n)) )  }
		else:
			nv = { n:np.mean( [ G.nodes[k][name] for k in G.neighbors(n) ] ) for n in G.nodes }
		nv[source_id] = source_val
		nx.set_node_attributes(G, nv, name)

	if noise:
		x = get_node_data(G,name=name) + np.random.normal(0,noise,size=len(G.nodes))
		set_node_data(G,x,name=name)


