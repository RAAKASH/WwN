from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython


cpdef dijkstra(int n,np.ndarray[np.float64_t, ndim=1] dist,np.ndarray[long, ndim=1] temp_parent,int s,int e, dict g,int verbose):
    dist[s] = 0
    cdef np.ndarray[int, ndim=1] fixed = np.array([0]*n)
    fixed[s]=1
    
    dijkstra_inner(n, fixed , dist,temp_parent,s,e, g, verbose)



@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef int dijkstra_inner(int n,np.ndarray[int, ndim=1] fixed,np.ndarray[np.float64_t, ndim=1] dist,np.ndarray[long, ndim=1] temp_parent,int s,int e, dict g,int verbose)except ? -1:
    
    
    
    
  
    cdef list chosen = [s]
    cdef int iter1=1
    while 1:
        for c in chosen:
            for j,cost in g[c].items():
               if dist[j]>dist[c]+cost:
                    dist[j]=dist[c]+cost
                    temp_parent[j] = c
                    
            if verbose>=2:
                print("Iteration:",iter1)
                print("chosen")
                iter1=iter1+1
                print(fixed,c)
        tmp = dist[fixed ==0]
        
        if len(tmp)==0:
            path = []
            break
        else:
            tmp_min = np.inf
            #for i in range(len(tmp)):
            #    if tmp_min>tmp[i]:
            #        k=i
                    
            k = np.min(tmp)
            #print(k)
            k = np.where(dist==k)
            k = k[0]
            #chosen = []
            for l in k:
                
                fixed[l]=1
                chosen[0]=l
                break
        
        if e in chosen:
            break



cpdef floyd_warshall(double D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt, int verbose=0):
    cdef int v = int(D_shp)
    #print(D,nxt)
    floyd_warshall_inner(v,D, nxt,verbose)
    return (D,nxt)


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef int floyd_warshall_inner(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt, int verbose)except ? -1:
    cdef int v = D_shp
    
    for k in range(v):
        if verbose>=1:
            print("\nStage:",k,"\n",D)
            
        for i in range(v):
            if i==k:
                continue
            for j in range(v):
                if j == k:
                    continue

                if D[i,j]>(D[i,k]+D[k,j]):
                    if verbose>=2:
                        print(" Changed",i,"->",j,"from ",D[i,j],"to ",D[i,k]+D[k,j])
                    D[i,j] = D[i,k]+D[k,j]
                    nxt[i,j]=nxt[i,k]
                        
                        
    
        if verbose>=2:
            input("Press Enter to continue")

cpdef DP(double D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[long, ndim=2] nxt,int mode, int verbose=0):
    cdef int v = int(D_shp)
    cdef np.ndarray[np.float64_t, ndim=2] W = D.copy()
    cdef np.ndarray[np.float64_t, ndim=2] W_copy = W.copy()
    if mode==0:
        W=DP_inner(v,D,W,W_copy, nxt,verbose)
    else:
        W = DP_inner_naive(v,D,W,W_copy, nxt,verbose)
    return (W,nxt)

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef np.ndarray[np.float64_t, ndim=2] DP_inner(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[np.float64_t, ndim=2] W,np.ndarray[np.float64_t, ndim=2] W_copy,np.ndarray[long, ndim=2] nxt, int verbose):
    cdef int v = D_shp
    
    
    for stage in range(v-2):
        W_copy = W.copy()
        if verbose>=1:
            print("\nStage:", stage,"\n",W)
           

        for i in range(v):
            for j in range(v):
                for k in range(v):
                
                    if W_copy[i,j]>(W[i,k]+W[k,j]):
                        if verbose>=1:
                            print(" Changed",i,"->",j,"from ",W[i,j],"to ",W[i,k]+W[k,j])
                     
                        W_copy[i,j] = W[i,k]+W[k,j]
                        nxt[i,j]=nxt[i,k]
        if verbose>=2:
                input("Press Enter to continue")
                
        if (W_copy==W).all():
            if verbose>=1:
                print("Breaking out, algorithm stagnant")
            break
        else:
            W = W_copy
    return W        
            
cdef DP_inner_naive(int D_shp,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[np.float64_t, ndim=2] W,np.ndarray[np.float64_t, ndim=2] W_copy,np.ndarray[long, ndim=2] nxt, int verbose):
    cdef int v = D_shp
    
    
    for stage in range(v-2):
        W_copy = W.copy()
        if verbose>=1:
            print("\nStage:", stage,"\n",W)
            

        for i in range(v):
            for j in range(v):
                for k in range(v):
                
                    if W_copy[i,j]>(W[i,k]+W[k,j]):
                        if verbose>=1:
                            print(" Changed",i,"->",j,"from ",W[i,j],"to ",W[i,k]+W[k,j])
                     
                        W_copy[i,j] = W[i,k]+D[k,j]
                        nxt[i,j]=nxt[i,k]
        if verbose>=2:
            input("Press Enter to continue")
        if (W_copy==W).all():
            print("Algorithm Constant, Breaking out")
            break
        else:
            W = W_copy
    return W