import numpy as np
import networkx as nx
from tqdm import tqdm
import xlwt
from xlwt import Workbook
import re
#from pyvis.network import Network
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from WwN.Algorithms import*
class graph:
    def __repr__(self):
        return (str(self.getgraph()))
    def __init__(self):
        node.i = 0
        self.node_list = []
        self.graph = {}
        self.arclist = []
        self.path_dfs = []
        self.parent_list = []
        self.temp_read_file_name = []
    def create_nodes(self, m):
        """ Create "m" number of nodes and returns the same
        """
        tmp = [node(num = i) for i in range(len(self.node_list),m+len(self.node_list))]
        self.node_list = self.node_list  +tmp
        self.parent_list = self.node_list.copy()
        return tmp

    def read_file(self,name,overwrite=0):
        n = len(self.node_list)
        if self.temp_read_file_name!=name or overwrite==1:
            #print("Reading file:",name)
            self.temp_read_file_name = name
            file1 = open(name, "r+")
            file1.seek(0)
            a = file1.readlines()
            a[0] = a[0].replace("\n", "")
            a[0] = a[0].split()
            if len(a[0])>2:
                 a[0] = [int(a[0][0])+n,int(a[0][1])+n,float(a[0][2]),float(a[0][3])]

            else:
                a[0] = list(map(int, a[0]))
          
            for i in range(1,len(a)):
                a[i] = a[i].replace("\n", "")
                a[i] = a[i].split()
                #a[i] = list(map(float, a[i]))
                a[i] = [int(a[i][0])+n,int(a[i][1])+n,float(a[i][2]),float(a[i][3])]

                #a[i] = [int(a[i][0]),int(a[i][1]),np.float(a[i][2]),np.float(a[i][3])]
            if len(a[0])>2:
                #print(a[-1][0])
                a = [[a[-1][0]+1]]+a
            self.add_arcs(a)
            return a
        else:
            print("File already in buffer, change overwrite parameter to reread the file")
    def add_arcs(self, n,method="num"):
        if isinstance(n, dict):
            for a in n.keys():
                a.addchild(n[a])
        else:
            if method=="num":
                self.create_nodes(n[0][0])
                #print(n[0])   
                for i in tqdm(range(1,len(n)),desc="Loading File:"+self.temp_read_file_name):
                    self.addarc(n[i])
            else:
                for i in tqdm(n,desc="Loading Arcs:"):
                    #print(i)
                    self.addarc(i,method)
                
        self.getarclist()
        self.getgraph()

    def get_arcinfo(self, a, b):
        a = self.node_list[a]
        b = self.node_list[b]
        return ([a.getcost(b), a.getcapacity(b)])

    def clear(self):
        self.i = 0
        self.node_list = []
        self.graph = {}
        self.arclist = []
        node.i = 0

    def addarc(self, arc,method="num"):
        #try:
        if method=="num":
            self.node_list[arc[0]].addchild(self.node_list[arc[1]], arc[2], arc[3])
        else:
            self.node_list[arc[0].num].addchild(self.node_list[arc[1].num], arc[2], arc[3])
           
        #except:
            #print(arc[0])
        #self.getarclist()

    def getarclist(self):
        self.arclist=[]
        for j in self.node_list:
            if j.getchild() == []:
                pass
            else:
                for k in j.getchild():
                    self.arclist.append([j, k, j.getcost(k), j.getcapacity(k)])
        return self.arclist

    def getgraph(self, type="non_adj"):
        """ Dicitonary based graph
        
        Input: type (optional)\n
        Output: Returns dictionary based graph (can be accessed using obj.graph)
        
        """
        if type == "adj":
            for j in self.node_list:
                self.graph[j] = j.getchild()

        else:
            for j in self.node_list:
                if j.getchild() == []:
                    pass
                else:
                    self.graph[j] = j.getchild()

        return self.graph
    def copy(self):
        gr = graph()
        gr.create_nodes(len(self.node_list))
        
        gr.add_arcs(self.arclist,method="nodes")
        return gr
    
    def create_graph(self,adj_mat):
        """
        Create graph from adjacency matrix
        Input : Adjacency matrix
        """
        self.clear()
        l,m = np.shape(adj_mat)
        self.create_nodes(m)
        arc = np.where(adj_mat !=np.inf)
        l,m  = np.shape(arc)

        for i in tqdm(range(m),desc="Loading graph"):
            if arc[0][i]!=arc[1][i]:
                self.node_list[arc[0][i]].addchild([self.node_list[arc[1][i]]],[adj_mat[arc[0][i],arc[1][i]]])
        self.getarclist()
    
    def create_randgraph(self, m, density=0.5,method=None):
        """ Creating Random graph for algo testing
        
        Input: No of nodes, density (optional)\n
        Output: Graph dictionary (can be accessed by obj.getgraph() as well)
        
        """
        obj = self.create_nodes(m)
        arr = np.where(np.random.rand(m, m) > (1 - density))
        cost = np.random.rand(m, m)*3
        l, m = np.shape(arr)
        for i in tqdm(range(m),desc="Loading rand graph"):
            if arr[0][i] == arr[1][i]:
                pass
            else:
                if method==None:
                    obj[arr[0][i]].addchild([obj[arr[1][i]]],[cost[arr[0][i],arr[1][i]]])
                else:
                    obj[arr[0][i]].addchild([obj[arr[1][i]]],[method(cost[arr[0][i],arr[1][i]])])
                    
        self.getarclist()
        return self.getgraph()

    def plot(self,size = 12, d=None,weight="cost"):
        """ Graph plotting
        
        Input: size of plot, dictionary based graph (optional), weight = "cost","capacity",or anything else \n
        Output: Planar/spring plot basis planarity condition
        """
        print("\n ********** In plotting function *********")
        if d == None:
            d = self.getgraph()

        g = nx.DiGraph()
        g.add_nodes_from(d.keys())
        if weight=="cost":
            for k, v in d.items():
                #g.add_edges_from(([(k, t) for t in v]))
                g.add_weighted_edges_from(([(k, t,k.getcost(t)) for t in v]))
        else:
            for k, v in d.items():
                #g.add_edges_from(([(k, t) for t in v]))
                g.add_weighted_edges_from(([(k, t,k.getcapacity(t)) for t in v]))
        
            
        plt.figure(figsize=(size,size))
        labels ={}
        if weight=="cost":
            for k, v in d.items():
                for t in v:
                    labels[(k, t)] =round(k.getcost(t),2)
            
                
        elif weight=="capacity":
            for k, v in d.items():
                for t in v:
                    labels[(k, t)] =round(k.getcapacity(t),2) 
        else:
           
            for k, v in d.items():
                for t in v:
                    labels[(k, t)] = str(round(k.getcost(t),2)) + "/"+ str(round(k.getcapacity(t),2)) 
            
        try:
            pos=nx.planar_layout(g) 
            nx.draw_networkx(g,pos)

            nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        except:
            print("Graph not planar changing to spring type")
            pos=nx.spring_layout(g) 
            nx.draw_networkx(g,pos)
            nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        plt.show()

    def dfs(self, s, e=None,verbose=0):
        """ 
        DFS Algorithm 
        
        Input: Start node, End node (optional), verbose =0,1,2 \n
        Output: (path (profs algo), visited list)
        """
        if verbose>=1:
            print("\n********** Performing Depth first search using DFS method ***********")
        
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e]
        self.visited = []
        
        path = []
        self.visited.append(s)
      
        path.append(s)
        self.path_dfs = []
        self.path_dfs_2 = []

        self.dfs_algo( path, s, e,verbose)
       
        if set(self.visited) == set(self.node_list):
            if verbose>=1:
                print("All nodes CAN be visited, returning path and visited nodes")
        else:
            if verbose>=1:
                print("All nodes CANT be visited, returning path and visited nodes")
            #print("Nodes that haven't be visited:" + str(set(self.node_list) - set(self.visited)))
        return (self.path_dfs_2,self.visited) #self.path_dfs

    def dfs_algo(self,  path, s, e,verbose=0):
        if verbose>=2:
            print("Visited nodes list:",self.visited)
        connected = [a for a in s.getchild() if a not in self.visited]
        self.visited = self.visited + connected
        for i in connected:
            i.temp_parent = s
            
            if e == i:
                # Returning Path using back tracking algorithm, self.path_dfs is actual path
                self.path_dfs = (path + [i])
                prt = e
                path = []
                while 1:
                    
                    if prt==self.visited[0]:
                        #print("In Loop",path,prt)
                        self.path_dfs_2 =  [prt]+path
                        break
                    else:
                        path =[prt]+path
                        
                        prt = prt.temp_parent
                break
            else:
                try:
                    self.dfs_algo( path + [i], i, e,verbose)
                except:
                    print(i)
                    print(self.dfs_algo( path, i, e))

    
    def search(self,s,e=None,method = "bfs",verbose=0,capacity=0):
        """
        DFS and BFS algorithm \n
        Inputs: input, output(optional), method = "bfs"/"dfs", verbose = 0,1,2; capacity=0/1 (0 if capacity not to be considered) \n
        Output: (Path, visited and unseen)
        """
        
        
        if verbose>=1:
            if method=="bfs":
                print("\n**********Performing BFS using Search method**********")
            else:
                print("\n**********Performing DFS using Search method**********")
            # Node checking
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e]
        
        active = [s];
        unseen = self.node_list.copy()
        unseen.remove(s)
        visited = [];
        vis_act = [0]*len(self.node_list)
        vis_act[s.num]=1
        while (active!=[]): 
 
            s = active[0]
            if verbose>=2:
                print("Node under inspection, Active and Unseen:",end="")
                print(s,active,unseen)
            visited.append(s)
            active.remove(s)
            
            # Path tracing algorithm if destination node found
            if s == e:
                if verbose>=1:
                    print("Path found, returing path, visited and unseen")
                prt = e
                path = []
                while 1:
                    if prt==visited[0]:
                        
                        return ([prt]+path,visited,unseen)
                    else:
                        path =[prt]+path
                        #ind = self.node_list.index(prt)
                        prt = prt.temp_parent
                   
                    
            
            #tmp = [a for a in s.getchild() if a not in visited and a not in active]
           
            if capacity==1:        
                tmp = [a for a in s.getchild() if vis_act[a.num]==0 and s.getcapacity(a)!=0]
            else:
                tmp = [a for a in s.getchild() if vis_act[a.num]==0]
            #tmp   = list(set(s.getchild()) - set(visited)-set(active)) # Same as checking child in unseen but faster
            
            # Adding temp parent for path tracing
            for i in tmp:
                i.temp_parent = s
                vis_act[i.num]=1
            #self.parent_list = np.array(self.parent_list)
            #self.parent_list[tmp] = s
            
            if method == "bfs":
                active =  active + tmp
               
            else:
                active = tmp + active
            
            
            unseen = list(set(unseen) - set(tmp))
            
        if verbose>=1:   
            print("No Path found, returning path,visited")
        return ([],visited)#, unseen)
        
    def mod_bell_ford(self,s,e,verbose=0):
        """ Modified Bellman Ford Algorithm
        
        Inputs: Start node, End node, verbose = 0,1,2 \n
        Output: Returns path ([] if no path), use obj.node_list[e].dist for finding  shortest distance
        """
        
        if verbose>=1:
            print("\n**********Modified Bellman-Ford algorithm**********")
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e] 
            
        
        for i in self.node_list:
            i.dist = np.inf
            i.temp_parent = None
        s.dist = 0
        change = True
        self.getarclist()
        iter1=0
        while(change ==True):
            
            change = False
            for i in self.arclist:
                if(i[1].dist>i[0].dist+i[0].getcost(i[1])):
                    i[1].dist = i[0].dist+i[0].getcost(i[1])
                    i[1].temp_parent = i[0]
                    change=True
            if verbose>=2:
                print("\n"+"Iteration:"+str(iter1+1))
                for i in self.node_list:
                    print(i.dist,end=" ")
                print("")
                for i in self.node_list:
                    print(i.temp_parent,end=" ")
            iter1=iter1+1;
            if iter1>= len(self.node_list)*2+1:
                print("")
                print("Negative loop found",e.dist)
                
                break
        if e.dist==np.inf:
            return []
            if verbose>=1:
                print("\nPath not found")
        else:
            iter1 = 0
            path = []
            if verbose>=1:
                print("\nPath found")
            prt = e
            tmp = len(self.node_list)*2
            while 1:
                if prt==s:
                    print([prt]+path)
                    return [prt]+path
                else:
                    path =[prt]+path
                    prt = prt.temp_parent
                iter1=iter1+1
                if iter1>tmp:
                    print("Negative Loop in path finding")
                    return []
    def bell_ford(self,s,e,verbose=1):
        """
        Input: s, e 
        Output: Returns shortest path
        refer to self.distmat and self.dist
        """
        if verbose>=1:
            print("\n***********Bellman-Ford algorithm***********")
        dist = [np.inf]*len(self.node_list)
        
        distmat = [dist]
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e] 
        
        for i in self.node_list:
            i.temp_parent=None
        dist[s.num]=0
        change = True
        self.getarclist()
        iter1=0
        
        while(change ==True):
            change = False
            dist_copy = dist.copy()
            for i in self.arclist:
                #Added additional capacity constraint to prevent 0 capacity arcs
                if i[0].getcapacity(i[1])>0:
                    if(dist[i[1].num] >dist[i[0].num]+i[0].getcost(i[1])):
                        dist_copy[i[1].num] = dist[i[0].num]+i[0].getcost(i[1])
                        i[1].temp_parent = i[0]
                        change=True
                distmat.append(dist_copy)
            
            dist = dist_copy
            if verbose>=2:
                print("\n"+"Iteration:"+str(iter1+1))
                print(dist_copy)
                for i in self.node_list:
                    print(i.temp_parent,end=" ")
            iter1=iter1+1;
            if iter1>=len(self.node_list)*2:
                break
        self.dist = dist
        self.distmat=distmat            
        if dist[e.num]==np.inf:
            print("path not found")
        else:
            path = []
            if verbose>=1:
                print("\nPath found")
            prt = e
            while 1:
                if prt==s:
                    return [prt]+path
                else:
                    path =[prt]+path
                    prt = prt.temp_parent
    
                    
    def dijkstra(self,s,e=None,verbose=0):
        """
        Output: Path and distance
        Use obj.path, obj.dist for accessing path and distance
        """
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e] 
            
        path = []
        dist = np.array([np.inf]*len(self.node_list))
        dist[s.num] = 0
        fixed = np.array([0]*len(self.node_list))
        fixed[s.num]=1
        chosen = [s]
        iter1=1
        while 1:
            for c in chosen:
                for j in c.getchild():
                   if dist[j.num]>dist[c.num]+c.getcost(j):
                        dist[j.num]=dist[c.num]+c.getcost(j)
                        j.temp_parent = c

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

                k = min(tmp)
                #print(k)
                k = np.where(dist==k)
                k = k[0]
                chosen = []
                for l in k:
                    fixed[l]=1
                    chosen.append( self.node_list[l])
                    
            
        if e!=None:         
            if dist[e.num]!=np.inf:
                if verbose>=1:
                    print("\nPath found")
                prt = e
                while 1:
                    
                    if prt==s:
                        path = [prt]+path
                        break
                    else:
                        #print(prt.num)
                        path =[prt]+path
                        prt = prt.temp_parent
        else:
            path=[]
        self.dist = dist
        self.path = path
        return (path,dist)
    
    def arbitrage(self,arbitrage_node=0,verbose=0):
        """ Arbitrage algorithm - Uses bellman ford
        
        Input: Arbitrage node, verbose
        """
        gr = self.copy()
        if isinstance(arbitrage_node, int): 
            arbitrage_node  = gr.node_list[arbitrage_node]
        
        tmp_arclist = np.array(gr.arclist)
        rout = tmp_arclist[tmp_arclist[:,1]==arbitrage_node,:]
        tmp_node = gr.create_nodes(1)
        rout[:,1] = tmp_node
        gr.add_arcs(rout,"node")
        #pth = gr.search(arbitrage_node,tmp_node[0])[0]
        pth = gr.mod_bell_ford(arbitrage_node,tmp_node[0],verbose)
        print("\nCost:",gr.get_path_cost(pth))
        return(pth,gr.get_path_cost(pth) )
        
    def writegraph(self,name="demofile.txt"):
        """
        Input=Name of file
        
        """
        
        f = open(name, "w+")
        f.write(str(len(self.node_list))+" " + str(len(self.arclist))+"\n")
        for i in tqdm(range(len(self.arclist))):
            f.write(str(self.arclist[i][0].num)+" "+str(self.arclist[i][1].num)+" "+str(self.arclist[i][2])+" "+str(self.arclist[i][3])+"\n")
        f.close()
    
     
    
    def replace_arc(self,s,e,cost,capacity=None):
        """ Replacing arc information
        
        Inputs: Tail node, Head node, cost, capacity (optional)
        """
        s.replacecost(e,cost)
        tmp = np.array(self.arclist)
        ind = np.where((tmp[:,0]==s) & (tmp[:,1]==e))[0][0]
        
        if capacity!=None:
            s.replacecapacity(e,capacity)
            self.arclist[ind][3] = capacity
        self.arclist[ind][2] = cost
       
        
    def replace_arcs(self,arclist):
        """ Replacing arc information
        
        Inputs: Tail node, Head node, cost, capacity (optional)
        """
        for i in arclist:
            i[0].replacecost(i[1],i[2])
            i[0].replacecapacity(i[1],i[3])
        self.arclist =arclist
    
    def get_path_cost(self,path):
        for i in range(len(path)):
            if isinstance(path[i], int):
                path[i] = self.node_list[path[i]]

        c = 0
        
        path[0].dist=0
        for i in range(len(path)-1):
            c=c+path[i].getcost(path[i+1])
            path[i+1].dist=c
        return c
    def get_adj_mat(self):
        """Adjacency Matrix - Intermediate Floyd Warshall
        
        Returns 0th iteration/stage adjacency matrix of Floyd Warshall algorithm
       
        """
        
        m  = len(self.node_list)
        self.adj_mat = np.zeros((m,m))+np.inf
        for i in self.arclist:
            self.adj_mat[i[0].num,i[1].num] = i[2]
        for i in self.node_list:
            self.adj_mat[i.num,i.num]=0
        return self.adj_mat
            
    def get_parent_mat(self,method="node"):
        """Parent Matrix - Intermediate Floyd Warshall
        
        Returns 0th iteration/stage heads of Floyd Warshall algorithm
       
        """
        if method=="node":
            mat = np.empty((len(self.node_list),len(self.node_list)), dtype=node, order='C')
    
            for i in self.node_list:
                mat[i.num,i.num] = i
            for i in self.arclist:
                mat[i[0].num,i[1].num]=i[1]
        else:
            mat = np.empty((len(self.node_list),len(self.node_list)),dtype=int)
    
            for i in self.node_list:
                mat[i.num,i.num] = i.num
            for i in self.arclist:
                mat[i[0].num,i[1].num]=i[1].num
            
        return mat
    
    def floyd_warshall_cyth(self,verbose=0):
        """  Floyd Warshall Algorithm - Cythonized to increase speeds by 100-400 times
        
        Output: Returns Shortest distance and heads (of shortest paths) from all nodes to all nodes\n
        Can be accessed through obj.D and obj.nxt
        """
        if verbose>=1:
            print("\n ***************Floyd Warshall Algorithm*******************")
        self.method="num"
        nxt = self.get_parent_mat(method=self.method)
            
        D = self.get_adj_mat()
        D,nxt = floyd_warshall(len(self.node_list),D,nxt,verbose)
        self.nxt = nxt
        self.D = D
        return(D,nxt)
    
    
    def DP_cyth(self,mode=0,verbose=0):
        """  DP Algorithm - Cythonized to increase speeds by 100-400 times
        Input: Mode - 0/1: 0 - Optimized, 1 - Non-optimized
        Output: Returns Shortest distance and heads (of shortest paths) from all nodes to all nodes\n
        Can be accessed through obj.D and obj.nxt
        """
        if verbose>=1:
            print("\n\n**************Dynamic programming routine, mode=",mode,"************************")
        self.method="num"
        nxt = self.get_parent_mat(method=self.method)
            
        D = self.get_adj_mat()
        D,nxt = DP(len(self.node_list),D,nxt,mode,verbose)
        self.nxt = nxt
        self.D = D
        return(D,nxt)
            
    def floyd_warshall_naive(self, verbose=0):
        """ Floyd Warshall Algorithm - Naive implementation     
        Output: Returns Shortest distance and heads (of shortest paths)
        from all nodes to all nodes\n
        Can be accessed through obj.D and obj.nxt
        """
        if verbose>=1:
            print("\n\n********* Floyd Warshall algorithm************")
        self.method = "node"
        nxt = self.get_parent_mat()
            
        D = self.get_adj_mat()
        for k in tqdm(self.node_list,"Stage"):
            if verbose>=1:
               print("\n Stage:",k.num,"\n",D)
            for i in self.node_list:
                if i==k:
                    continue
                for j in self.node_list:
                    if j == k:
                        continue
                    n = i.num
                    m = j.num
                    o = k.num
                    if D[n,m]>(D[n,o]+D[o,m]):
                        if verbose>=1:
                            print(" Changed",n,"->",m,"from ",D[n,m],"to ",D[n,o]+D[o,m])
                        D[n,m] = D[n,o]+D[o,m]
                        nxt[n,m]=nxt[n,o]

            
            if verbose>=2:
                input("Press Enter to continue")
        self.nxt = nxt
        self.D = D
        return(D,nxt)
    def return_path(self,u,v,method=None):
        """  Floyd Warshall Algorithm - complementary\n
        Requires Floyd Warshall algorithm to be run first
        
        Input: Start nodes, End nodes \n
        Output: Returns Shortest path 
        """
        if method==None:
            method = self.method
        
        if isinstance(u, int):
            u = self.node_list[u]
        if isinstance(v, int):
            v = self.node_list[v]
        if self.nxt[u.num,v.num] == None:
            return []
        if method=="node":
            path = [u]
            while u != v:
                u  = self.nxt[u.num,v.num]
                path.append(u)
        else:
            path = [u]
            while u != v:
                u  = self.node_list[self.nxt[int(u.num),int(v.num)]]
                path.append(u)
            
        return path
    
    
    def second_best_path(self,pth):
        """  Returns the second best path\n
             Note: It necessarily returns the 2nd best path
        Input: Start nodes, End nodes \n
        Output: Returns Shortest path, cost. Refer to obj.secondary_paths to get full list of paths
        """
        self.get_path_cost(pth) 
        min_dist = pth[-1].dist
        tmp_path=[]
        tmp_cost=np.inf
        secondary_paths = []
        for i in range(len(pth)):
            if pth[i] == pth[-1]:
                break
            tmp = pth[i].getcost(pth[i+1])
            tmp_cost1 = pth[i].dist
            
            self.replace_arc(pth[i],pth[i+1],np.inf)
            pth1 = self.mod_bell_ford(pth[i],pth[-1])
            
            if tmp_cost>pth[-1].dist+tmp_cost1 and min_dist < pth[-1].dist+tmp_cost1:
                tmp_cost = pth[-1].dist+tmp_cost1
                tmp_path = pth[0:i]+pth1
                secondary_paths.append(tmp_path)
                print(tmp_cost,tmp_path)
            else:
                print(pth[-1].dist+tmp_cost1,pth[0:i]+pth1)
                secondary_paths.append(pth[0:i]+pth1)
            
            self.replace_arc(pth[i],pth[i+1],tmp)
            self.secondary_paths = secondary_paths
            self.get_path_cost(pth)
        return (tmp_path,tmp_cost)
        

    
    def Astar(self,s,e,D=np.array([0]),verbose=0):
        
        """
        A* Algorithm \n
        Input: s (Start node), e (end node), D = Distance function - (Astar, Dijkstras),verbose=0/1/2 
        Output: Shortest path
        """
        if (D==0).all():
            D=np.zeros((len(self.node_list),len(self.node_list)))
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e] 
            
        path = []
        dist = np.array([np.inf]*len(self.node_list)) # distance from source node
        dist[s.num]=0
        tmp_nodelist = self.node_list.copy()           
        chosen = []
        c=s
       
        iter1=1
        ind =0
        dist1 = D[:,e.num].T # Shortest distance to end node
        dist2 = np.array([0.0]*len(self.node_list)) # Chosen array
        tmp_dist=np.array([0])
        
        while tmp_nodelist!=[] and not (tmp_dist==np.inf).all():
            chosen.append(c)
            tmp_nodelist.remove(c)
            for i in c.getchild():
                if dist[i.num]>dist[c.num] + c.getcost(i):
                    dist[i.num] = dist[c.num] + c.getcost(i)
                    i.temp_parent=c
                #print(i,c.getcost(i))
            
            dist2[c.num] = np.inf            
            tmp_dist = dist1+dist+dist2

            #for d in chosen:
            #    tmp_dist[d.num] = np.inf
            
            ind  = np.argmin(tmp_dist) 
            #self.node_list[ind].temp_parent= c
            c = self.node_list[ind]    
                
            if verbose>=1:
                print("\nIteration:",iter1)
                iter1=iter1+1
                print("Nodes fixed:",chosen)
                print("Distances fixed:",dist+dist1)
                print("Distances fixed:",tmp_dist)
            
              
            if chosen[-1]==e:
                break
                
            
             
        if dist[e.num]!=np.inf:
            if verbose>=1:
                print("\nPath found")
            prt = e
            while 1:
                
                if prt==s:
                    path = [prt]+path
                    break
                else:
                    #print(path)
                    path =[prt]+path
                    prt = prt.temp_parent
                
        self.dist = dist
        self.path = path
        return (path,dist)
 
    def binarysearch(self,func,max_val=100,min_val=0,tol=10**-3,verbose=0):
        """ Binary Search \n
        Input : Function to evaluate, max_val, min_val, tol
        
        
        """
        if verbose>=1:
            print("\n************ Binary Search ****************")
            print("Max value:",max_val,", Min value:",min_val,"\n")
        while  abs(max_val-min_val)>tol:
            cur_pt = (max_val+min_val)/2
            if verbose>=1:
                print("Evaluating Point:",cur_pt)
            if func(cur_pt)>0:
                min_val = cur_pt
                cur_pt = (cur_pt+max_val)/2
            else:
                max_val = cur_pt
                cur_pt = (cur_pt+min_val)/2
        return max_val
    
    def optimal_loop(self,g):
        """ Function used for finding the optimal loop basis benefit vs cost
        
        Benefit in 3rd column, cost in 4th column \n
        Use the function obj.binarysearch(obj.optimal_loop,max_val,min_val)
        """
        tmp1 = np.array(self.arclist)
        tmp = tmp1.copy()
        tmp[:,2] =-tmp1[:,2]+g*tmp1[:,3]
        self.replace_arcs(tmp)
        """
        g2 = graph()
        g2.create_nodes(len(self.node_list))
        g2.add_arcs(tmp,"node")
        """
        self.floyd_warshall_cyth()
        flag = (np.diag(self.D)<0).any()
        self.replace_arcs(tmp1)
        if flag:
            return 1
        else:
            return -1
        
    
    def capacity_compare(self,gr):
        flow = []
        for i in self.arclist:
            cap =  i[3]- gr.node_list[i[0].num].getcapacity(gr.node_list[i[1].num])
            if cap>0:
                flow.append([i[0],i[1],cap])
        return flow
                
        
    def create_residual_network(self):
        
        gr1 = self.copy()
        for i in gr1.arclist:
            if i[0] in i[1].getchild():
                pass
            else:
                gr1.addarc([i[1],i[0],-i[2],0],method="node")#-i[2]
        for i in gr1.node_list:
            i.sort()
        gr1.getarclist()
        return gr1
        
    def ford_fulkerson(self,s,t,verbose=0):
        """ Ford fulkerson/Edmonds Karp implementation using bfs
        
        Output: 1) Residual graph: paths which can be accessed through obj.flow_var
                2) Total flow
                3) Flow for all arcs - Access through obj.flow
        """
        if verbose>=1:
            print("\n********Ford-Fulkerson***********")
            print("Returns residual graph, use obj.flow_var for different paths, obj.flow_val for maximum flow value, obj.flow for flows across different arcs \n\n")
        gr = self.create_residual_network();
        flow_val = 0
        if isinstance(s,int):
           s = gr.node_list[s]
        if isinstance(t,int):
           t = gr.node_list[t]        
        
        path = gr.search(s,t,capacity=1)[0]
        self.flow_var=[]
        if path==[]:
            print("No path found")
            return 0
        while path!=[]:
            mincap = np.inf
            for i in range(len(path)-1):
                mincap =min(mincap,path[i].getcapacity(path[i+1]))
            
            if verbose>=1:
                print("Path found",path,"Minimum cap:",mincap)
            self.flow_var += [path,mincap]

            flow_val+=mincap
            for i in range(len(path)-1):
                gr.replace_arc(path[i],path[i+1],path[i].getcost(path[i+1]),path[i].getcapacity(path[i+1])-mincap) 
                gr.replace_arc(path[i+1],path[i],path[i+1].getcost(path[i]),path[i+1].getcapacity(path[i])+mincap)
            
            path = gr.search(s,t,capacity=1)[0]
            self.flow = self.capacity_compare(gr)
            self.flow_val = flow_val
        return (gr,flow_val,self.flow)
        
    def set_distance_lables(self,t,gr=None):
        """
            Creates new graph along with distance labels
        """
        if gr==None:
            gr = graph()
            gr.create_nodes(len(self.node_list))
            for i in self.arclist:
                gr.addarc([i[1].num,i[0].num,1,i[3]])
            gr.getarclist()
           
            
        gr.dijkstra(t)
        return(gr)
        
    def push(self,active,s,t,verbose=0):
        flag=0
        tmp = active.copy()
        for v in tmp:
            f=0
            for j in v.getchild():
                if self.dist[j.num]<self.dist[v.num] and v.getcapacity(j)>0:
                    
                    f = min(v.excess,v.getcapacity(j))
                    flag=1
                    v.excess -=f
                    j.excess +=f
                    
                    if verbose>=1:
                        print("\nPushing volume ",round(f,2),"from ",v,"to ",j)
                        if verbose>=2:
                            print("Left over excess at ",v," is ",round(v.excess,2))
                    
                    if j !=s and j!=t:
                        if j not in active:
                            if active!=[]:
                                if verbose>=3:
                                    print("adding",j)
                                f1 = 0
                                for i in range(len(active)):
                                
                                    if active[i].num>j.num:
                                        active.insert(i,j)
                                        f1=1
                                        break
                                    
                                if f1==0:
                                    active.append(j)
                            else:
                                active.append(j)
                                
                    
                    if verbose>=3:
                        print("Active list:",active)         
                    self.replace_arc(v,j,v.getcost(j),v.getcapacity(j)-f)
                    self.replace_arc(j,v,j.getcost(v),j.getcapacity(v)+f)
                    
                    
                 
                if v.excess ==0:
                    active.remove(v)
                    
                    if verbose>=2:
                        print("Removing ",v," from active list")
                        if verbose>=3:
                            print("Active list:",active) 
                    break
                if f>0:
                    break
                    
            if flag==1:
                break
            
        return flag
        
            
            
    def relabel(self,active,verbose=0):
        min_val=np.inf
        if active==[]:
            return 0
        for j in active:
            #print(j)
            if min_val>self.dist[j.num]:
                v=j
                min_val = self.dist[j.num]
                
        
        w = min([self.dist[j.num] for j in v.getchild() if v.getcapacity(j)>0] )   
        if verbose>=1:
            #print("Active list:", active," Distance:",self.dist)
            print("\nRelabelling ",v," distance to ",w+1)
            #print(v.getchild(),v.excess)
        self.dist[v.num] = w+1
        
        
    def push_relabel(self,s,t,verbose=0):
        """
        Input: Start node, end node
        
        Output: Residual network graph, maxflow value
        obj.flow can be used for getting flow through each arc
        
        """
        if verbose>=1:
            print("\n***********Push Relabel algo running**************")
        gr1 = self.create_residual_network()
        gr_tmp  = self.set_distance_lables(t)
        gr1.dist = gr_tmp.dist
        if verbose>=1:
            print('Inital Distance list: ',gr_tmp.dist)
        if isinstance(s, int):
            s = gr1.node_list[s]
        if isinstance(t, int):
            t = gr1.node_list[t]
        #dist = gr1.dijkstra(t)[1]
        
        
        if gr1.dist[s.num]==np.inf:
            print("No path between start and end nodes")
            return 0
        gr1.dist[s.num] =  sum(np.array(gr1.dist)<np.inf)
        active =[]
        for i in gr1.node_list:
            i.excess = 0
        
        # Preflow
        if verbose>=1:
            print("\n*****Initial flow*****")
        for j in s.getchild():
            if j!=t:
                active.append(j)
            j.excess = s.getcapacity(j)
            gr1.replace_arc(s,j,s.getcost(j),0)
            gr1.replace_arc(j,s,j.getcost(s),j.excess)
            if verbose>=1:
                
                print("Pushing volume ",j.excess,"from ",s,"to ",j)
        iter1=1
        if verbose>=1:
            
            print("\n*******Iteration starts******")
        while active!=[]:
            if gr1.push(active,s,t,verbose) == 0:
                gr1.relabel(active,verbose=verbose)
            
            if verbose>=1:
                print("\n******Iteration ", iter1," Complete******\n", "Flow achieved ", t.excess,"\n")
            iter1+=1
            
        self.flow = self.capacity_compare(gr1)   
        self.flow_val = t.excess
        return gr1,t.excess
        

    def capacity_scaling(self,s,t,f,verbose=0):
        """
        Finds least cost for flowing "f" units of flow 
        """
        if verbose>=1:
            print("******* Running Capacity Scaling Algorithm*******")
        gr = self.create_residual_network()
        if isinstance(s,int):
           s = gr.node_list[s]
        if isinstance(t,int):
           t = gr.node_list[t]
           
        flow_cost = 0
        remaining = f
        path = gr.bell_ford(s,t,0)
        if verbose>=1:
            print("Initial Path: ",path)
         
        iter1 = 0
        while path!=[] and remaining>0:
            iter1+=1
            if verbose>=1:
                print("*****Iteration:",iter1,"******")
                
            c = remaining
            #c=np.inf
            for i in range(len(path)-1):
                c =min(c,path[i].getcapacity(path[i+1]))
                
            if verbose>=1:
                print("Remaining:",remaining)
                print("Minimum capacity:",c)
                print("Remaining:",remaining-c)
            remaining-= c
            flow_cost+= c*gr.dist[t.num]
            
            if verbose>=1:
                print("Minimum Cost:",gr.dist[t.num])
        
            
            for i in range(len(path)-1):
                 gr.replace_arc(path[i],path[i+1],path[i].getcost(path[i+1]),path[i].getcapacity(path[i+1])-c) 
                 gr.replace_arc(path[i+1],path[i],path[i+1].getcost(path[i]),path[i+1].getcapacity(path[i])+c)
               
            if remaining > 0:
                path = gr.bell_ford(s,t,0)
                if verbose>=1:
                    print("Path found:",path)
        if remaining == 0:
            flow = self.capacity_compare(gr)
            if verbose>=1:
                print("Paths found, Full flow complete, returning flow and flow cost")
                print("Flow:",flow,"\n","Flow Cost:",flow_cost)
            return flow,flow_cost
        else:
            
            print("Can not send ",f," units")
            return [],np.inf
                
                
    def maxflow_mincut(self,s,t,method=0,name="temp.xls",verbose=1):
        """
        Note: Use this only if absolutely necessary else use your brain, you might get faster results

        Parameters
        ----------
        s : Int or node
            start node.
        t : Int or node
            End node.
        method : int, optional
            DESCRIPTION. The default is 0.
        name : String, optional
            DESCRIPTION. Name of excel file The default is "temp.xls".
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        if isinstance(s,int):
           s = self.node_list[s]
        if isinstance(t,int):
           t = self.node_list[t]    
    
        wb = Workbook()
  
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('Sheet 1')
        sheet1.write(0,0,'Arcs')
        sheet1.write(0,1,'Capacity')
        iter1=1
        for i in self.arclist:
            sheet1.write(iter1,0,str(i[0].num)+"-"+str(i[1].num))
            sheet1.write(iter1,1,i[3])
            iter1+=1;
        
        
        iter1=5
        
        style1 = xlwt.easyxf('pattern: pattern solid, fore_colour yellow;')
        style2 = xlwt.easyxf('pattern: pattern solid, fore_colour blue;')
        style3 = xlwt.easyxf('pattern: pattern solid, fore_colour orange;')
        for i in self.node_list:
            sheet1.write(0,iter1,"z_"+str(i.num))
            sheet1.write(1,iter1,0)
            sheet1.write(2,iter1,0,style1)
            iter1+=1
            
        sheet1.write(1,iter1,xlwt.Formula("TRANSPOSE(B2:B"+str(len(self.arclist)+1)+")"))
        for i in self.arclist:
            sheet1.write(0,iter1,"y_"+str(i[0].num)+str(i[1].num))
            sheet1.write(2,iter1,0,style1)
            iter1+=1
            
        sheet1.write(0,iter1,"y_ts")
        sheet1.write(2,iter1,0,style1)
        sheet1.write(1,iter1,10**5)
        sheet1.write(2,4,"variables")
        sheet1.write(4,4,"Objective")
        try:
            sheet1.write(4,5,xlwt.Formula("SUMPRODUCT(F2:"+chr(70+len(self.arclist)+len(self.node_list))+"2,F3:"+chr(70+len(self.arclist)+len(self.node_list))+"3)"),style2)
        except:
            print("Update the formula manually in excel")
            
        sheet1.write(5,4,"Constraints")
        
        iter2=1
        for i in self.arclist:
            sheet1.write(5+iter2,4,str(i[0].num) +"-"+str(i[1].num))
            sheet1.write(5+iter2,5+i[0].num,1)
            sheet1.write(5+iter2,5+i[1].num,-1)
            sheet1.write(5+iter2,4+len(self.node_list)+iter2,1)
            try:
                sheet1.write(5+iter2,iter1+2,xlwt.Formula("SUMPRODUCT(F"+str(5+iter2+1)+":"+chr(70+len(self.arclist)+len(self.node_list))+str(5+iter2+1)+",F3:"+chr(70+len(self.arclist)+len(self.node_list))+"3)"),style3)
            except:
                print("Enter the formula manually in excel")
            sheet1.write(5+iter2,iter1+3,".GE.",style3)
            sheet1.write(5+iter2,iter1+4,0,style3)
            iter2+=1
        sheet1.write(5+iter2,4, "t-s")
        sheet1.write(5+iter2,5+s.num,-1)
        sheet1.write(5+iter2,5+t.num,1)
        sheet1.write(5+iter2,4+len(self.node_list)+iter2,1)
        try:
            sheet1.write(5+iter2,iter1+2,xlwt.Formula("SUMPRODUCT(F"+str(5+iter2+1)+":"+chr(70+len(self.arclist)+len(self.node_list))+str(5+iter2+1)+",F3:"+chr(70+len(self.arclist)+len(self.node_list))+"3)"),style3)
        except:
            print("Enter the formula manually in excel")
        sheet1.write(5+iter2,iter1+3,".GE.",style3)
        sheet1.write(5+iter2,iter1+4,1,style3)
        wb.save("Files/"+name)
    
class node:
    i = 0

    def __init__(self, parent=[], child=[], cost=[], capacity=[],num=i):
        self.num = num
        self.__parent = []
        self.__child = []
        self.__cost = []
        self.__capacity = []
        self.name = "N" + str(node.i)
        node.i = node.i + 1
        self.addparent(parent, cost, capacity)
        self.addchild(child)
        self.dist = np.inf

    def __repr__(self):
        self.name = "N" + str(self.num)
        return self.name
    
    
    
    def addparent(self, parent, cost=[], capacity=[]):
        if cost == []:
            cost = np.zeros(len(parent))
        if capacity == []:
            capacity = np.zeros(len(parent))

        if isinstance(parent, node):
            if parent in self.__parent:
                pass
            else:
                self.__parent.append(parent)

                parent.__child.append(self)
                parent.__cost.append(cost)
                parent.__capacity.append(capacity)

        elif isinstance(parent, list):
            for l in parent:
                if isinstance(l, node):
                    if l in self.__parent:
                        pass
                    else:
                        self.__parent.append(l)

                        l.__cost.append(cost[parent.index(l)])
                        l.__capacity.append(capacity[parent.index(l)])
                        l.__child.append(self)

    def getcost(self, d=None):
        """ Getting Cost of arc
        
        Inputs: Head of arc \n
        Output: Cost
        """
        if d==None:
            return(self.__cost)
        if d in self.__child:
            return self.__cost[self.__child.index(d)]
        else:
            return np.inf
    
    def replacecost(self,d,cost):
        self.__cost[self.__child.index(d)] = cost
        
    def replacecapacity(self,d,capacity):
        self.__capacity[self.__child.index(d)] = capacity

    def getcapacity(self, d=None):
        """ Getting Capacity of arc
        
        Inputs: Head of arc \n
        Output: Capacity
        """
        if d==None:
            return self.__capacity
        if d in self.__child:
            return self.__capacity[self.__child.index(d)]
        else:
            return 0
    def addchild(self, child, cost=[], capacity=[]):
        """ Adding a child
        
        Inputs: child/children, cost (optional), capacity (optional)\n
        Output: Child added, parent updated
        
        """
        if cost is []:
            cost = np.zeros(len(child))
        if capacity is []:
            capacity = np.zeros(len(child))
        if isinstance(child, node):
            if child in self.__child:
                pass
            else:
                child.__parent.append(self)

                self.__child.append(child)
                self.__cost.append(cost)
                self.__capacity.append(capacity)


        elif isinstance(child, list):
            for l in child:
                if l in self.__child:
                    pass
                else:
                    if isinstance(l, node):
                        l.__parent.append(self)

                        self.__child.append(l)
                        self.__cost.append(cost[child.index(l)])
                        self.__capacity.append(cost[child.index(l)])

    def getparent(self):
        return self.__parent

    def getchild(self):
        return self.__child

    def sort(self):
        c= self.__child.copy()        
        cost = self.__cost.copy()
        cap = self.__capacity.copy()
        c_num = [i.num for i in c ]
        c_sort = np.argsort(c_num)
        for i in range(len(c)):
            self.__child[i] = c[c_sort[i]]
            self.__cost[i] = cost[c_sort[i]]
            self.__capacity[i] = cap[c_sort[i]]            
def timecalc(func):
    def wrap(*args, **kwargs):
        t1 = timer()
        result = func(*args,**kwargs)
        t2 = timer()
        try:
            if kwargs["verbose"]=="on":
                print(result)
        except:
            print(result)
            

        return t2-t1
        
    return wrap  


if __name__ == "__main__":
   
    gr = graph()
    gr.clear()
    gr.read_file("Data/Quiz-4.txt")
    #gr.read_file("Data/q4-3.txt")
    #gr.read_file("Data/network_2.txt")
    #gr.read_file('Data/Class_DP_ExNet.txt')
    #gr.read_file("Data/Edmonds_Karp_weak.txt")
    #gr.read_file("Data/ford_fulk_weak.txt")
    #gr.read_file("Data/Optimal_Loop.txt")
    #gr.create_randgraph(100,0.5)
    #gr1=gr.ford_fulkerson(0,11)
    """
    gr.create_nodes(6)
    gr.addarc([0,1,1,7])
    gr.addarc([0,3,1,1])
    gr.addarc([1,2,1,4])
    gr.addarc([1,4,1,2])
    gr.addarc([2,5,1,6])
    gr.addarc([3,4,1,1])
    gr.addarc([4,5,1,9])
    gr.getarclist()
    """
    
    
    
    """
    gr.create_nodes(8)
    gr.addarc([0,1,4,3])
    gr.addarc([0,3,3,4])
    gr.addarc([0,4,5,6])
    gr.addarc([1,2,4,2])
    gr.addarc([1,6,6,5])
    gr.addarc([2,7,3,4])
    gr.addarc([3,1,2,3])
    gr.addarc([4,2,2,3])
    gr.addarc([4,3,2,4])
    gr.addarc([4,5,7,5])
    gr.addarc([5,7,1,2])
    gr.addarc([6,2,1,5])
    gr.addarc([6,7,1,4])
    gr.getarclist()
    gr.capacity_scaling(0,7,6,1)
    """
    """
    gr.create_nodes(8)
    gr.addarc([0,1,1,10])
    gr.addarc([1,2,1,2])
    gr.addarc([1,3,1,2])
    gr.addarc([1,4,1,1])
    gr.addarc([1,5,1,1])
    gr.addarc([2,6,1,1])
    gr.addarc([3,6,1,1])
    gr.addarc([4,6,1,1])
    gr.addarc([5,6,1,2])
    gr.addarc([6,7,1,10])
    
    gr.getarclist()
    
    gr2 = gr.push_relabel(0,7,verbose=2)
    """
    #gr.read_file("Data/Quiz3.txt")
    #gr.read_file("Data/Quiz3-Part2.txt")
    #a = gr.push_relabel(0,7,verbose=3)
    """
    gr.create_nodes(9)
    gr.addarc([0,1,-3,1])
    gr.addarc([0,2,-6,1])
    gr.addarc([0,3,-2,1])
    gr.addarc([1,4,-5,1])
    gr.addarc([3,5,-2,1])
    gr.addarc([1,6,-7,1])
    gr.addarc([4,7,-4,1])
    gr.addarc([5,7,-4,1])
    gr.addarc([2,7,-4,1])
    gr.addarc([7,8,0,1])
    gr.addarc([6,8,0,1])
    gr.getarclist()
    
    print(gr.mod_bell_ford(0,8))
    """
    """
    gr.create_nodes(12)
    gr.addarc([0,1,1,1.5])
    gr.addarc([0,2,1,1.2])
    gr.addarc([0,3,1,1.5])
    gr.addarc([0,4,1,2.6])
    gr.addarc([0,5,1,2.1])    
    gr.addarc([1,5+2,1,10])
    gr.addarc([1,5+3,1,10])
    gr.addarc([2,5+4,1,10])
    gr.addarc([2,5+5,1,10])
    gr.addarc([3,5+3,1,10])
    gr.addarc([3,5+4,1,10])
    gr.addarc([4,5+1,1,10])
    gr.addarc([4,5+2,1,10])
    gr.addarc([4,5+3,1,10])
    gr.addarc([4,5+4,1,10])
    gr.addarc([5,5+2,1,10])
    gr.addarc([5,5+3,1,10])
    gr.addarc([5+1,11,1,1])
    gr.addarc([5+2,11,1,2])
    gr.addarc([5+3,11,1,2])
    gr.addarc([5+4,11,1,3])
    gr.addarc([5+5,11,1,1])
    gr.getarclist()
    #gr.writegraph("Data/Work_schedule.txt")
    """
    
    

    #gr.plot(weight=1)
    #gr.writegraph(name="Data/ford_fulk_weak.txt")
    #gr.read_file("Data/Q2data.txt")
    #gr.read_file("Data/Optimal_Loop.txt")
    #gr.create_randgraph(100,0.1)
    #gr.read_file("Data/OldenburgEdges_Pro.txt",overwrite=1)
    
    #gr.floyd_warshall_cyth()
    #tmp_adj = gr.D#+np.around((np.random.rand(1000,1000)-0.5)*0.2,2);
    #tmp_adj[:,:]=0
    #np.fill_diagonal(tmp_adj,0)
    #gr.Astar(0,7,tmp_adj,verbose=0)
    
    """
    gr.create_nodes(10)
    gr.addarc([0,1,1,1000])
    gr.addarc([1,2,1,1000/6])
    gr.addarc([1,3,1,1000/6])
    gr.addarc([1,4,1,1000/6])
    gr.addarc([1,5,1,1000/6])
    gr.addarc([1,6,1,1000/6])
    gr.addarc([1,7,1,1000/6])
    gr.addarc([2,8,1,1000/6])
    gr.addarc([3,8,1,1000/6])
    gr.addarc([4,8,1,1000/6])
    gr.addarc([5,8,1,1000/6])
    gr.addarc([6,8,1,1000/6])
    gr.addarc([7,8,1,1000/6])
    gr.addarc([8,9,1,1000])
    gr.getarclist()
    """
    
    def f(g=10):
        if g>0.1823:
            return -1
        else:
            return 1
    #print("\n",gr.binarysearch(f,tol=10**-6))
    """
    def f(g=10,gr=gr,size=len(gr.node_list)):
        tmp = np.array(gr.arclist)
        
        tmp[:,2] =-tmp[:,2]+g*tmp[:,3]
        g2 = graph()
        g2.create_nodes(size)
        g2.add_arcs(tmp,"node")
        g2.floyd_warshall_cyth()
        #print(np.diag(g2.D))
        if (np.diag(g2.D)<0).any():
            return 1
        else:
            return -1
    """
    
    #print(f())
    #print("\n",gr.binarysearch(gr.optimal_loop,max_val=1000,min_val=0,tol=10**-3,verbose=1))
    #gr.Astar(0,3,(gr.D),verbose=2)
    #gr.read_file('Data/Class_DP_ExNet.txt')
    #gr.read_file('Data/Optimal_Loop.txt')
    #gr.read_file("Data/neg_no_cycle.txt")
    #gr.read_file("Data/neg_cycle.txt")
    #gr.read_file("Data/neg_no_cycle.txt")
    #gr.floyd_warshall_naive(1)
    #gr.return_path(0,1)
    #gr.read_file("Data/ExNet.txt")
    #gr.read_file("Data/InClass-2.txt")
    #gr.read_file("Data/Arbit.txt")
    
    #gr.create_randgraph(4,1,np.log)
    #gr.create_randgraph(300,0.1)
    #gr1 = gr.copy()

    #"""
    t = timecalc(gr.floyd_warshall_cyth)
    t1 = timecalc(gr.floyd_warshall_naive)
    t2 = timecalc(gr.mod_bell_ford)
    t3 = timecalc(gr.return_path)
    t4 = timecalc(gr.dijkstra)
    t5 = timecalc(gr.DP_cyth)
    t6 = timecalc(gr.Astar)
    #t()
    #"""
    #gr.create_randgraph(500,0.01)
    #pth = gr.mod_bell_ford(0,7)
    #pth1 = gr.second_best_path(pth)
    #t()
    
    """ 
     7/7/21 Class code 
    """
    """
    gr.read_file("Data/InClass-2.txt")
    print("\n")
    pth = gr.mod_bell_ford(0,7) #optimal path
    
    
        
    tmp_path=[]
    tmp_cost=np.inf
    
    for i in range(len(pth)):
        if pth[i] == pth[-1]:
            break
        tmp = pth[i].getcost(pth[i+1])
        tmp_cost1 = pth[i].dist
        #print("Shortest",pth[-1].dist-pth[i].dist)
        
        gr.replace_arc(pth[i],pth[i+1],np.inf)
        #print(pth[i].getcost(pth[i+1]))
        pth1 = gr.mod_bell_ford(pth[i],pth[i+1])
        #print("New Path",pth[-1].dist)
        
        if tmp_cost>pth[-1].dist+tmp_cost1:
            #print(pth1[-1].dist,pth[-1].dist)
            #print(pth[-1].dist,tmp_cost1,pth[i],i)
            tmp_cost = pth[-1].dist+tmp_cost1
            tmp_path = pth[0:i]+pth1
            print(tmp_cost,tmp_path)
        else:
            print(pth[-1].dist+tmp_cost1,pth[0:i]+pth1)
           
        
        gr.replace_arc(pth[i],pth[i+1],tmp)
        gr.mod_bell_ford(0,pth[-1])
        
    #gr.create_randgraph(10,0.5)
    
    """
    """
    arbitrage_node = 3
    arbitrage_node  = gr.node_list[arbitrage_node]
    
    tmp_arclist = np.array(gr.arclist)
    rout = tmp_arclist[tmp_arclist[:,1]==arbitrage_node,:]
    rout[:,1] = gr.create_nodes(1)
    gr.add_arcs(rout,"node")
    pth = gr.search(arbitrage_node,4)[0]
    print("\nCost:",gr.get_path_cost(pth))
    """
    #print(gr.arbitrage(1))
    """
    Check tramp steamer problem, code this shit
    
    """