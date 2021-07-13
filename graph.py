import numpy as np
import networkx as nx
from tqdm import tqdm
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
        
        if self.temp_read_file_name!=name or overwrite==1:
            #print("Reading file:",name)
            self.temp_read_file_name = name
            file1 = open(name, "r+")
            file1.seek(0)
            a = file1.readlines()
            a[0] = a[0].replace("\n", "")
            a[0] = a[0].split(" ")
            a[0] = list(map(int, a[0]))

            for i in range(1,len(a)):
                a[i] = a[i].replace("\n", "")
                a[i] = a[i].split(" ")
                #a[i] = list(map(float, a[i]))
                a[i] = [int(a[i][0]),int(a[i][1]),float(a[i][2]),float(a[i][3])]

                #a[i] = [int(a[i][0]),int(a[i][1]),np.float(a[i][2]),np.float(a[i][3])]

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
                for i in tqdm(range(len(n)),desc="Loading File:"+self.temp_read_file_name):
                    if i == 0:
                        self.create_nodes(n[i][0])
                    else:
                        self.addarc(n[i])
            else:
                for i in tqdm(n,desc="Loading Arcs:"):
                    
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

    def draw(self,size = 12, d=None):
        """ Graph plotting
        
        Input: size of plot, dictionary based graph (optional) \n
        Output: Planar/spring plot basis planarity condition
        """
        print("\n ********** In plotting function *********")
        if d == None:
            d = self.getgraph()

        g = nx.DiGraph()
        g.add_nodes_from(d.keys())
        for k, v in d.items():
            #g.add_edges_from(([(k, t) for t in v]))
            g.add_weighted_edges_from(([(k, t,k.getcost(t)) for t in v]))
        
        plt.figure(figsize=(size,size))
        
        try:
            pos=nx.planar_layout(g) 
            nx.draw_networkx(g,pos)
            labels = nx.get_edge_attributes(g,'weight')
            nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        except:
            print("Graph not planar changing to spring type")
            pos=nx.spring_layout(g) 
            nx.draw_networkx(g,pos)
            labels = nx.get_edge_attributes(g,'weight')
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

    
    def search(self,s,e=None,method = "bfs",verbose=0):
        """
        DFS and BFS algorithm \n
        Inputs: input, output(optional), method = "bfs"/"dfs", verbose = 0,1,2 \n
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
            iter1=iter1+1;
            if iter1>= len(self.node_list)+1:
                print("Negative loop found",e.dist)
                break
        if e.dist==np.inf:
            return []
            if verbose>=1:
                print("\nPath not found")
        else:
            
            path = []
            if verbose>=1:
                print("\nPath found")
            prt = e
            while 1:
                if prt==s:
                    print([prt]+path)
                    return [prt]+path
                else:
                    path =[prt]+path
                    prt = prt.temp_parent
                    
    def bell_ford(self,s,e,verbose=0):
        """
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
            
        dist[s.num]=0
        change = True
        self.getarclist()
        iter1=0
        
        while(change ==True):
            change = False
            dist_copy = dist.copy()
            for i in self.arclist:
                if(dist[i[1].num] >dist[i[0].num]+i[0].getcost(i[1])):
                    dist_copy[i[1].num] = dist[i[0].num]+i[0].getcost(i[1])
                    i[1].temp_parent = i[0]
                    change=True
                distmat.append(dist_copy)
            
            dist = dist_copy
            if verbose>=2:
                print("\n"+"Iteration:"+str(iter1+1))
                print(dist_copy)
            iter1=iter1+1;
            
        self.dist = dist
        self.distmat=distmat            
        if dist[e.num]==np.inf:
            print("path not found")
        else:
            path = []
            print("\nPath found")
            prt = e
            while 1:
                if prt==s:
                    return [prt]+path
                else:
                    path =[prt]+path
                    prt = prt.temp_parent
    
                    
    def dijkstra(self,s,e,verbose=0):
        """
        Output: Path and distance
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
        while fixed[e.num]==0:
            for c in chosen:
                for j in c.getchild():
                    if dist[j.num]>dist[c.num]+c.getcost(j):
                        dist[j.num]=dist[c.num]+c.getcost(j)
                        j.temp_parent = c

                tmp = dist[fixed ==0]
                if verbose>=2:
                    print("Iteration:",iter1)
                    iter1=iter1+1
                    print(dist)
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
                
        self.dist = dist
        self.path = path
        return (path,dist)
    
    def arbitrage(self,arbitrage_node=0,verbose=0):
        
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
        if capacity!=None:
            s.replacecapacity(e,capacity)
        tmp = np.array(self.arclist)
        ind = np.where((tmp[:,0]==s) & (tmp[:,1]==e))[0][0]
        self.arclist[ind][2] = cost
        self.arclist[ind][3] = capacity
    
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
            print("*******************Dynamic programming routine, mode=",mode,"************************")
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
            print("********* Floyd Warshall algorithm************")
        self.method = "node"
        nxt = self.get_parent_mat()
            
        D = self.get_adj_mat()
        for k in tqdm(self.node_list,"Stage"):
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
                        D[n,m] = D[n,o]+D[o,m]
                        nxt[n,m]=nxt[n,o]
            if verbose>=1:
               print("\n Stage:",k.num,"\n",D)
               if verbose>=2:
                   input("Press Enter to continue")
        self.nxt = nxt
        self.D = D
        return(D,nxt)
    def return_path(self,u,v):
        """  Floyd Warshall Algorithm - complementary\n
        Requires Floyd Warshall algorithm to be run first
        
        Input: Start nodes, End nodes \n
        Output: Returns Shortest path 
        """
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

    def getcost(self, d):
        """ Getting Cost of arc
        
        Inputs: Head of arc \n
        Output: Cost
        """
        if d in self.__child:
            return self.__cost[self.__child.index(d)]
        else:
            return np.inf
    
    def replacecost(self,d,cost):
        self.__cost[self.__child.index(d)] = cost
        
    def replacecapacity(self,d,capacity):
        self.__capacity[self.__child.index(d)] = capacity

    def getcapacity(self, d):
        """ Getting Capacity of arc
        
        Inputs: Head of arc \n
        Output: Capacity
        """
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
    #gr.read_file("Data/neg_no_cycle.txt")
    #gr.read_file("Data/neg_cycle.txt")
    #gr.read_file("Data/neg_no_cycle.txt")
    #gr.floyd_warshall_naive(1)
    #gr.return_path(0,1)
    #gr.read_file("Data/ExNet.txt")
    #gr.read_file("Data/InClass-2.txt")
    #gr.read_file("Data/Arbit.txt")
    
    #gr.create_randgraph(4,1,np.log)
    gr.create_randgraph(300,0.1)
    #gr1 = gr.copy()

    #"""
    t = timecalc(gr.floyd_warshall_cyth)
    t1 = timecalc(gr.floyd_warshall_naive)
    t2 = timecalc(gr.mod_bell_ford)
    t3 = timecalc(gr.return_path)
    t4 = timecalc(gr.dijkstra)
    t5 = timecalc(gr.DP_cyth)
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