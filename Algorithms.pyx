from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython



cpdef floyd_warshall(double D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt):
    cdef int v = int(D_shp)
    #print(D,nxt)
    floyd_warshall_inner(v,D, nxt)
    return (D,nxt)

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef int floyd_warshall_inner(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt)except ? -1:
    cdef int v = D_shp
    
    for k in range(v):
            for i in range(v):
                if i==k:
                    continue
                for j in range(v):
                    if j == k:
                        continue

                    if D[i,j]>(D[i,k]+D[k,j]):
                        D[i,j] = D[i,k]+D[k,j]
                        nxt[i,j]=nxt[i,k]
    
    

cpdef DP(double D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt,int mode):
    cdef int v = int(D_shp)
    cdef np.ndarray[np.float64_t, ndim=2] W = D.copy()
    cdef np.ndarray[np.float64_t, ndim=2] W_copy = W.copy()
    if mode==0:
        W=DP_inner(v,D,W,W_copy, nxt)
    else:
        W = DP_inner_naive(v,D,W,W_copy, nxt)
    return (W,nxt)

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef np.ndarray[np.float64_t, ndim=2] DP_inner(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[np.float64_t, ndim=2] W,np.ndarray[np.float64_t, ndim=2] W_copy,np.ndarray[long, ndim=2] nxt):
    cdef int v = D_shp
    
    
    for stage in range(v-2):
        W_copy = W.copy()
        #print(stage)
        for i in range(v):
            for j in range(v):
                for k in range(v):
                
                    if W_copy[i,j]>(W[i,k]+W[k,j]):
                        W_copy[i,j] = W[i,k]+W[k,j]
                        nxt[i,j]=nxt[i,k]
        if (W_copy==W).all():
            #print(stage)
            break
        else:
            W = W_copy
    return W        
            
cdef int DP_inner_naive(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[np.float64_t, ndim=2] W,np.ndarray[np.float64_t, ndim=2] W_copy,np.ndarray[long, ndim=2] nxt):
    cdef int v = D_shp
    
    
    for stage in range(v-2):
        W_copy = W.copy()
        #print(stage)
        for i in range(v):
            for j in range(v):
                for k in range(v):
                
                    if W_copy[i,j]>(W[i,k]+W[k,j]):
                        W_copy[i,j] = W[i,k]+D[k,j]
                        nxt[i,j]=nxt[i,k]
        if (W_copy==W).all():
            #print(stage)
            break
        else:
            W = W_copy
    return W