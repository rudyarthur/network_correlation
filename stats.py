import networkx as nx
import numpy as np

##############################################
##Add and extract data from NetworkX Objects##
##############################################

def get_node_data(G, name="data"):
	return np.array( [ G.nodes[k][name] for k in G.nodes ] )

def set_node_data(G, x, name="data"):
	nx.set_node_attributes(G, {n:x[i] for i,n in enumerate(G.nodes)}, name)


def row_normalise(A, loop=0):
	rowsums =  np.sum(A , axis=1) 
	for i in range( len(rowsums) ): A[i,:] /= (float(rowsums[i])+loop)


def get_adjacency(G, rownorm = True, loop=0):
	A = nx.adjacency_matrix(G).astype(float)			
	if rownorm: row_normalise(A,loop)
	return A


##############################################
## Permutation tests
##############################################
	
def data_permutations(x, Np):
	return [ np.random.permutation(x) for _ in range(Np) ]


#Compute the p-value of r given an estimate of the distribution rs = [ estimate1, estimate2, ... ]	
#Frequently we use 999 samples and +1 smoothing factor	
def pval(rs, r, alt="greater", smooth=0):
	larger = (np.array(rs) >= r).sum()
	if alt == "two-tailed" and (len(rs) - larger) < larger: larger = len(rs) - larger
	return (larger + smooth) / (len(rs) + smooth)
	

#####################
#Global Moran index
#####################
def compute_moran(A, x):
	z = (x - x.mean())	
	zl = A.dot(z) 
	return (z.shape[0] / A.sum()) * ( (z * zl).sum() ) / (z**2).sum()    

def moran_data_dist(A, x, Np):
	return np.array([ compute_moran(A, xp) for xp in data_permutations(x, Np) ])

def moran_config_dist(A, x, deg_seq, Np):
	vals = []
	for i in range(Np):
		Gc = nx.configuration_model( deg_seq )
		vals.append( compute_moran(get_adjacency(Gc), x) )
	return vals
	
def moran_pval(G, A, x, null="dist", Np=1000, alt="greater", smooth=0):
	I = compute_moran(A, x)
	if null == "dist":
		deg_seq = [d for n,d in G.degree]		
		dists = moran_config_dist(A,x,deg_seq,Np)
	elif null == "config":
		dists = moran_data_dist(A,x,Np)
	else:
		return I
			
	return I, pval(dists, I, alt=alt, smooth=smooth), dists

def moran(G, name="data", null="dist", Np=1000, alt="greater", smooth=0):
	A = get_adjacency(G)
	x = get_node_data(G, name=name) 
	return moran_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth)
	

#####################
##Local Moran index
#####################
def compute_local_moran(A, x, norm=False):
	z = (x - x.mean())	
	zl = A.dot(z) 

	if norm: return (z * zl) / (z**2).sum()    
	return (z * zl)   #no real need to normalise

def compute_local_moran_i(A, z, i):
	return z[i] * A[i,:].dot(z)[0]
	
def conditional_random(x,i): #keep i fixed, shuffle rest
	N = len(x)
	idx = list(np.random.permutation( np.concatenate( (np.arange(i), np.arange(i+1,N)) ) ) ); 
	idx.insert(i,i)
	return x[idx]
	
def local_moran_data_dist(A, x, Np=100):
	N = len(x)
	dists = np.zeros( (Np,N) )
	z = x - np.mean(x)
	for i in range(N):
		for j in range(Np):
			dists[j,i] = compute_local_moran_i( A, conditional_random(z,i) , i)
	return dists

def local_moran_config_dist(A, x, deg_seq, Np=100):
	N = len(x)
	dists = np.zeros( (Np,N) )
	for i in range(Np):
		Gc = nx.configuration_model( deg_seq )
		dists[i,:] = compute_local_moran(get_adjacency(Gc), x)
	return dists
	
	
def local_moran_pval(G, A, x, null="dist", Np=100, alt="greater", smooth=0):
	L = compute_local_moran(A, x)
	if null == "dist":
		deg_seq = [d for n,d in G.degree]		
		dists = local_moran_config_dist(A,x,deg_seq,Np)
	elif null == "config":
		dists = local_moran_data_dist(A,x,Np)
	else:
		return L
		
	return L, np.array([ pval(dists[i], L[i], alt=alt, smooth=smooth) for i in range(len(x)) ]), dists

def local_moran(G, name="data", null="dist", Np=1000, alt="greater", smooth=0):
	A = get_adjacency(G)
	x = get_node_data(G, name=name) 
	return local_moran_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth)
	
	
#################
#Lee statistic
################
def compute_lee(A,x,y,loop=None):
	zx = x - np.mean(x)
	zy = y - np.mean(y)

	zlx = A.dot(zx) 
	zly = A.dot(zy) 
		
	if loop is not None:
		rowsums =  np.asarray( np.sum(A , axis=1) ).reshape( (-1,) )
		zlx = (zlx + loop*zx)/(rowsums+loop)
		zly = (zly + loop*zy)/(rowsums+loop)	
		W = np.sum( np.sum(A, axis=1).reshape(rowsums.shape)/(rowsums+loop) ) 
	else:
		W = np.sum( np.sum(A, axis=0) ) 

	return (x.shape[0] / W) * ( (zlx * zly).sum() ) / np.sqrt( np.sum(zx**2) * np.sum(zy**2) )
	

def lee_pval(G, A, x, y, null="data", Np=100, alt="greater", smooth=0):
	r = compute_lee(A, x,y)

	rs = []
	if null == "data":
		xdist = data_permutations(x,Np)
		ydist = data_permutations(y,Np)
		for xp in xdist:
			for yp in ydist:
				rs.append( compute_lee(A, xp,yp) )		
	else:
		deg_seq = [d for n,d in G.degree]		
		for i in range(Np):
			Gc = nx.configuration_model( deg_seq )
			rs.append(  compute_lee(get_adjacency(Gc), x, y) )
	
	return r, pval(rs, r, alt=alt, smooth=smooth), rs

def lee(G, xname, yname, null="data", Np=100, alt="greater", smooth=0):
	A = get_adjacency(G)
	x = get_node_data(G, name=xname) 
	y = get_node_data(G, name=yname) 
	return lee_pval(G, A, x, y, null=null, Np=Np, alt=alt, smooth=smooth)
	


######################
#Pearson correlation	
######################

def compute_pearson(x,y):
	zx = x - np.mean(x)
	zy = y - np.mean(y)
	return zx.dot(zy)/np.sqrt( np.sum(zx**2) * np.sum(zy**2) )


def pearson_pval(x, y, Np=100, alt="greater", smooth=0):
	r = compute_pearson(x,y)

	rs = []

	xdist = data_permutations(x,Np)
	ydist = data_permutations(y,Np)
	for xp in xdist:
		for yp in ydist:
			rs.append( compute_pearson(xp,yp) )		
	
	return r, pval(rs, r, alt=alt, smooth=smooth), rs

def pearson(G, xname, yname, Np=100, alt="greater", smooth=0):
	x = get_node_data(G, name=xname) 
	y = get_node_data(G, name=yname) 
	return pearson_pval(x, y, Np=Np, alt=alt, smooth=smooth)
	
