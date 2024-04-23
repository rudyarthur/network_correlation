import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


##############################################
##Add and extract data from NetworkX Objects##
##############################################

def get_node_data(G, name="data"):
	return np.array( [ G.nodes[k][name] for k in G.nodes ] )

def set_node_data(G, x, name="data"):
	nx.set_node_attributes(G, {n:x[i] for i,n in enumerate(G.nodes)}, name)

def copy_node_data(Gsource, Gtarget):
	for k in Gsource.nodes[0]:
		if k != "community":
			set_node_data( Gtarget, get_node_data(Gsource, name=k), name=k )

def bin_node_data(G, bins, binval="midpt", name="data"):
	x = get_node_data(G,name=name)
	inds = np.digitize(x, bins)

	if binval == "left":	
		for i in range(1,len(bins)):	
			x[inds == i] = bins[i-1]
	elif binval == "right":
		for i in range(1,len(bins)):	
			x[inds == i] = bins[i]
	elif binval == "midpt":
		for i in range(1,len(bins)):	
			x[inds == i] = 0.5*(bins[i-1] + bins[i]) 
			
	set_node_data(G, x, name=name)

 
def node_data_astype(G, t, name="data"):
	x = get_node_data(G,name=name).astype(t)
	set_node_data(G, x, name=name)

def scale_node_data(G, s, name="data"):
	x = get_node_data(G,name=name) * s
	set_node_data(G, x, name=name)


#def row_normalise(A, loop=0):
#	rowsums =  np.sum(A , axis=1) 
#	for i in range( len(rowsums) ): A[i,:] /= (float(rowsums[i])+loop)  ##crazy slow
#	return A	
def row_normalise(A, loop=0):
	rowsums =  np.asarray( np.sum(A , axis=1) ).reshape( (-1,) ) + loop
	rowsums[ rowsums == 0 ] = 1 #zero rows stay zero rows
	rowsums = 1/rowsums
	
	D = csr_matrix( (rowsums, (np.arange(rowsums.shape[0]), np.arange(rowsums.shape[0]) ))  )
	return D.dot(A) 


def get_adjacency(G, rownorm = True, loop=0, drop_weights=False):
	A = nx.adjacency_matrix(G).astype(float)	
	if drop_weights: A[A > 0] = 1
	if rownorm: return row_normalise(A,loop)
	return A



##############################################
## Permutation tests
##############################################
	
def data_permutations(x, Np):
	return [ np.random.permutation(x) for _ in range(Np) ]


#Compute the p-value of r given an estimate of the distribution rs = [ estimate1, estimate2, ... ]	
#Sometimes 999 samples and +1 smoothing factor	
def pval(rs, r, alt="greater", smooth=0):

	m=1
	if alt == "lesser": m=-1
	larger = (m*np.array(rs) >= m*r).sum()
	if alt == "two-tailed" and (len(rs) - larger) < larger: larger = len(rs) - larger
	return (larger + smooth) / (len(rs) + smooth)
	

#####################
#Global Moran/Geary/Getis-Ord index
#####################

##Don't use networkx version becuase it's VERY inefficient
#sum_ij xi xj A_ij - sum_ij xi xj \sum_k w_ik \sum_l w_lj
#sum(x.Ax) 
#sum_i xi sum_k wik = sum_ik xi wik = sum_k (xA) 
#sum(x.Ax) - (sum(xA) * sum(Ax))
#ai = sum_j w_ij
#sum_i xi xi sum_j w_ij - (sum_ij xi w_ij)^2
#sum_i xi xi sum_j w_ij - (sum_ij xi w_ij)^2
#sum_j (x2A) - (sum_j xA)^2
#this is almost... the same as moran index!
def compute_assortativity(A,x):
	W = A.sum()

	x2 = x*x
	xl = A.dot(x) 
	x2l = A.dot(x2) 
	xr = A.transpose().dot(x) 	
	x2r = A.transpose().dot(x2) 	
	num = ((x*xl).sum()/W - (xl.sum()/W)*(xr.sum()/W)) 
	den = np.sqrt( (x2l.sum()/W - (xl.sum()/W)**2) * (x2r.sum()/W - (xr.sum()/W)**2) ) 
	
	return num/den
	
	
def compute_moran(A, x):
	z = (x - x.mean())	
	zl = A.dot(z) 
	return (z.shape[0] / A.sum()) * ( (z * zl).sum() ) / (z**2).sum()    

#def compute_moran(G): return compute_moran( get_adjacency(G), get_node_data(G) )
	
#geary
#sum_ij wij (xi - xj) (xi - xj)
#sum_ij wij (xi^2 - 2xi xj + xj^2)
#sum_i xi^2 wij + sum_j wij xj xj - 2sum_ij xi wij xj
def compute_geary(A, x):
	z = (x - x.mean())	
	x2 = x*x
	xl = A.dot(x) 
	x2l = A.dot(x2) 
	x2r = A.transpose().dot(x2) 	
	return ( (x.shape[0]-1) / (2*A.sum()) ) * (   x2l.sum() + x2r.sum() - 2*x.dot(xl) ) / (z**2).sum()    

#def compute_geary(G): return compute_geary( get_adjacency(G), get_node_data(G) )
	
#general GO 
def compute_getisord(A, x):
	xl = A.dot(x) 
	return  (x* xl).sum() / x.sum()**2
	
#def compute_getisord(G): return compute_getisord( get_adjacency(G), get_node_data(G) )
	
	
def compute_joincount(A, x):
	N = x.shape[0]//2 #bundled two vectors together to respect calling conventions
	xr = x[:N]
	xl = A.dot(x[N:]) 
	return  (xr * xl).sum() 


#sum_{hi | dhi = d} (x_h - x_i)(y_h - y_i)
#sum_{hi} w_{hi} (x_h - x_i)(y_h - y_i)
#sum_hi xh whi yh + sum_hi w_hi xi xi - sum_ih xi whi yh - sum_hi yi whi xh
#
def compute_crossvar(A, x):
	N = x.shape[0]//2 #bundled two vectors together to respect calling conventions

	a = x[:N]
	b = x[N:]

	ab = a*b

	al = A.dot(a) 
	bl = A.dot(b) 
	abl = A.dot(ab) 
	abr = A.transpose().dot(ab) 	
	return ( 1 / (2*A.sum()) ) * (   abl.sum() + abr.sum() - a.dot(bl) - b.dot(al) )   


def global_data_dist(A, x, Np, func=compute_moran):
	return np.array([ func(A, xp) for xp in data_permutations(x, Np) ])

def global_config_dist(G, A, x, Np, func=compute_moran):
	vals = []

	if G.is_directed():
		in_deg_seq = [d for n,d in G.in_degree]		
		out_deg_seq = [d for n,d in G.out_degree]		
		for i in range(Np):
			Gc = nx.directed_configuration_model( in_deg_seq , out_deg_seq )
			vals.append( func(get_adjacency(Gc), x) )

	else:
		deg_seq = [d for n,d in G.degree]		
		for i in range(Np):
			Gc = nx.configuration_model( deg_seq )
			vals.append( func(get_adjacency(Gc), x) )

	
			
	return vals
	
def global_pval(G, A, x, null="data", Np=1000, alt="greater", smooth=0, func=compute_moran, return_dists=True):
	I = func(A, x)
	if null == "config":
		dists = global_config_dist(G,A,x,Np,func)
	elif null == "data":
		dists = global_data_dist(A,x,Np,func)
	else:
		return I

	if return_dists:
		return I, pval(dists, I, alt=alt, smooth=smooth), dists
	return I, pval(dists, I, alt=alt, smooth=smooth)

def moran(G, name="data", null="data", Np=1000, alt="greater", smooth=0, rownorm=True, return_dists=True):
	A = get_adjacency(G, rownorm)
	x = get_node_data(G, name=name) 
	return global_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, return_dists=return_dists)
	
def geary(G, name="data", null="data", Np=1000, alt="lesser", smooth=0, rownorm=True, return_dists=True):
	A = get_adjacency(G, rownorm)
	x = get_node_data(G, name=name) 
	return global_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, func=compute_geary, return_dists=return_dists)

def getisord(G, name="data", null="data", Np=1000, alt="greater", smooth=0, rownorm=True, return_dists=True):
	A = get_adjacency(G, rownorm)
	x = get_node_data(G, name=name) 
	return global_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, func=compute_getisord, return_dists=return_dists)

def assortativity(G, name="data", null="data", Np=1000, alt="greater", smooth=0, rownorm=True, return_dists=True):
	A = get_adjacency(G, rownorm)
	x = get_node_data(G, name=name) 
	return global_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, func=compute_assortativity, return_dists=return_dists)


def joincount(G, r, s, name="data", null="data", Np=1000, alt="greater", smooth=0, return_dists=True):
	A = get_adjacency(G, rownorm=False, drop_weights=True)
	x = get_node_data(G, name=name) 
	xr = np.where( x==r, 1, 0)
	xs = np.where( x==s, 1, 0)
	y = np.concatenate( (xr,xs) )
	if r == s: y // 2

	return global_pval(G, A, y, null=null, Np=Np, alt=alt, smooth=smooth, func=compute_joincount, return_dists=return_dists)


##TODO, fix?
#def crossvar(G, name1, name2, null="data", Np=1000, alt="greater", smooth=0, return_dists=True):
#	A = get_adjacency(G, rownorm=False, drop_weights=True)
#	a = get_node_data(G, name=name1) 
#	b = get_node_data(G, name=name2) 
#
#	y = np.concatenate( (a,b) )
#	if name1 == name2: y // 2
#
#	return global_pval(G, A, y, null=null, Np=Np, alt=alt, smooth=smooth, func=compute_crossvar, return_dists=return_dists)


#####################
##Local Moran index
#####################
def compute_local_moran(A, x, norm=False):
	z = (x - x.mean())	
	zl = A.dot(z) 

	if norm: return (z * zl) / (z**2).sum()    
	return (z * zl)   #no real need to normalise

#local geary sum_j w_ij (xi - xj)**2
#= sum_j xi xi w_ij + wij xj xj - 2 xi wij xj
#= xi^2 * w_i + x2r - 2x*xl
def compute_local_geary(A, x, norm=False):
	x2 = x*x
	xl = A.dot(x) 
	x2l = A.dot(x2) 

	w =  x2*np.asarray( np.sum(A , axis=1) ).reshape( (-1,) )
	
	if norm: 
		z = (x - x.mean())	
		return (   x2l + w - 2*x*xl ) / (z**2).sum()    
	return (   x2l + w - 2*x*xl )   #no real need to normalise

#this is Gstar
def compute_local_getisordstar(A, x, norm = False):	
	if norm:
		return A.dot(x) /x.sum()
	return A.dot(x)
	
#this is G[i] = sum_{i=/=j} wij xj/sum{i=/=j} xj
# sum{i=/=j} xj = sum{j} xj - x[i]
#sum_{i=/=j} wij xj = sum_{j} wij xj - wii xi
def compute_local_getisord(A, x, norm=False):
	if norm:	
		return (A.dot(x) - A.diagonal()*x) / (np.ones(x.shape[0])*x.sum() - x)
	return A.dot(x) - A.diagonal()*x

def compute_local_moran_i(A, z, i):
	return z[i] * A[i,:].dot(z)[0]

def compute_local_geary_i(A, x, x2, i):
	return  x[i]*x[i]*A[i,:].sum() + A[i,:].dot(x2)[0]  -2*x[i] * A[i,:].dot(x)[0]

def compute_local_getisordstar_i(A, x, i):
	return  x[i] * A[i,:].dot(x)[0]
		
def compute_local_getisord_i(A, x, i):
	return  x[i] * A[i,:].dot(x)[0] - A[i,i]*x[i]

		
def conditional_random(x,i): #keep i fixed, shuffle rest
	N = len(x)
	idx = list(np.random.permutation( np.concatenate( (np.arange(i), np.arange(i+1,N)) ) ) ); 
	idx.insert(i,i)
	return x[idx]
	
def local_data_dist(A, x, Np=100, stat="moran"):
	N = len(x)
	dists = np.zeros( (Np,N) )
	if stat == "moran":
		z = x - np.mean(x)
		for i in range(N):
			for j in range(Np):
				dists[j,i] = compute_local_moran_i( A, conditional_random(z,i) , i)
	elif stat == "geary":
		for i in range(N):
			for j in range(Np):
				y = conditional_random(x,i)
				y2 = y*y
				dists[j,i] = compute_local_geary_i( A, y, y2, i)
	elif stat == "getisord":
		for i in range(N):
			for j in range(Np):
				dists[j,i] = compute_local_getisord_i( A, conditional_random(x,i), i)
	elif stat == "getisord*":
		for i in range(N):
			for j in range(Np):
				dists[j,i] = compute_local_getisordstar_i( A, conditional_random(x,i), i)

	return dists

def local_config_dist(G, A, x, Np=100, stat="moran"):
	N = len(x)
	dists = np.zeros( (Np,N) )
	
	if G.is_directed():
		in_deg_seq = [d for n,d in G.in_degree]		
		out_deg_seq = [d for n,d in G.out_degree]		
	else:
		deg_seq = [d for n,d in G.degree]		



	for i in range(Np):
		if G.is_directed():
			Gc = nx.directed_configuration_model( in_deg_seq , out_deg_seq )
		else:
			Gc = nx.configuration_model( deg_seq )
			
		if stat == "moran":
			dists[i,:] = compute_local_moran(get_adjacency(Gc), x)
		elif stat == "geary":
			dists[i,:] = compute_local_geary(get_adjacency(Gc), x)
		elif stat == "getisord":
			dists[i,:] = compute_local_getisord(get_adjacency(Gc), x)
		elif stat == "getisord*":
			dists[i,:] = compute_local_getisordstar(get_adjacency(Gc), x)

	return dists
	
	
def local_pval(G, A, x, null="data", Np=100, alt="greater", smooth=0, stat="moran"):

	if stat == "moran":
		L = compute_local_moran(A, x)
	elif stat == "geary":
		L = compute_local_geary(A, x)
	elif stat == "getisord":
		L = compute_local_getisord(A, x)
	elif stat == "getisord*":
		L = compute_local_getisordstar(A, x)
		
	if null == "config":
		dists = local_config_dist(G,A,x,Np,stat)
	elif null == "data":
		dists = local_data_dist(A,x,Np,stat)
	else:
		return L
		
	return L, np.array([ pval(dists[i], L[i], alt=alt, smooth=smooth) for i in range(len(x)) ]), dists

def local_moran(G, name="data", null="data", Np=1000, alt="greater", smooth=0):
	A = get_adjacency(G)
	x = get_node_data(G, name=name) 
	return local_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth)

def local_geary(G, name="data", null="data", Np=1000, alt="lesser", smooth=0):
	A = get_adjacency(G)
	x = get_node_data(G, name=name) 
	return local_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, stat="geary")

def local_getisord(G, name="data", null="data", Np=1000, alt="greater", smooth=0, star=True):
	A = get_adjacency(G)
	x = get_node_data(G, name=name) 
	if star:
		return local_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, stat="getisord*")
	return local_pval(G, A, x, null=null, Np=Np, alt=alt, smooth=smooth, stat="getisord")
			
	
#################
#Lee statistic
################
def compute_lee(A,x,y,loop=None):
	zx = x - np.mean(x)
	zy = y - np.mean(y)

	zlx = A.dot(zx) 
	zly = A.dot(zy) 
		
	if loop is not None:
		rowsums =  np.asarray( np.sum(A , axis=1) ).reshape( (-1,) ) + loop
		zlx = (zlx + loop*zx)/(rowsums)
		zly = (zly + loop*zy)/(rowsums)	
		W = np.sum( np.sum(A, axis=1).reshape(rowsums.shape)/(rowsums) ) 
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
		if G.is_directed():
			in_deg_seq = [d for n,d in G.in_degree]		
			out_deg_seq = [d for n,d in G.out_degree]		
			for i in range(Np):
				Gc = nx.directed_configuration_model( in_deg_seq , out_deg_seq )
				rs.append(  compute_lee(get_adjacency(Gc), x, y) )

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
	
