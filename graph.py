import numpy as np
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
import matplotlib.pyplot as plt
class graph:

    def __init__(self):
        node.i = 0
        self.node_list = []
        self.graph = {}
        self.arclist = []
        self.path_dfs = []
        self.parent_list = []
        self.temp_read_file_name = []
    def create_nodes(self, m):
        tmp = [node() for i in range(m)]
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
    def add_arcs(self, n):
        if isinstance(n, dict):
            for a in n.keys():
                a.addchild(n[a])
        else:
            for i in tqdm(range(len(n)),desc="Loading File:"+self.temp_read_file_name):
                if i == 0:
                    self.create_nodes(n[i][0])
                else:
                    self.addarc(n[i])
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

    def addarc(self, arc):
        self.node_list[arc[0]].addchild(self.node_list[arc[1]], arc[2], arc[3])
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

    def create_randgraph(self, m, density=0.5):
        obj = self.create_nodes(m)
        arr = np.where(np.random.rand(m, m) > (1 - density))
        l, m = np.shape(arr)
        for i in tqdm(range(m),desc="Loading rand graph"):
            if arr[0][i] == arr[1][i]:
                pass
            else:
                obj[arr[0][i]].addchild([obj[arr[1][i]]])
        return self.getgraph()

    def draw(self,size = 12, d=None):
        print("\n ********** In plotting function *********")
        net = Network(notebook=True)
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
            print("Change plot function type to spring")
        plt.show()
        #nx.draw_planar(g, with_labels=True)
        #nx.draw_spring(g, with_labels=True)
        #print(g)
        #net.from_nx(g)
        #net.show("example.html")
        
    def dfs(self, s, e=None,verbose="off"):
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
            print("All nodes CAN be visited, returning visited nodes and paths")
        else:
            print("All nodes CANT be visited, returning visited nodes and paths")
            #print("Nodes that haven't be visited:" + str(set(self.node_list) - set(self.visited)))
        return (self.visited, self.path_dfs, self.path_dfs_2)

    def dfs_algo(self,  path, s, e,verbose="off"):
        if verbose=="on":
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

        
    def search(self,s,e=None,method = "bfs",verbose="off"):
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
            if verbose=="on":
                print("Node under inspection, Active and Unseen:",end="")
                print(s,active,unseen)
            visited.append(s)
            active.remove(s)
            
            # Path tracing algorithm if destination node found
            if s == e:
                print("Path found, returing path")
                prt = e
                path = []
                while 1:
                    if prt==visited[0]:
                        
                        return [prt]+path
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
            
            
        print("No Path found, returning visited and unseen")
        return (visited, unseen)
        
    def mod_bell_ford(self,s,e,verbose="on"):
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
            if verbose=="on":
                print("\n"+"Iteration:"+str(iter1+1))
            for i in self.node_list:
                print(i.dist,end=" ")
            iter1=iter1+1;
                    
        if e.dist==np.inf:
            print("\nPath not found")
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
                    
    def bell_ford(self,s,e,verbose="on"):
        #dist_matrix  = list(np.full((1,len(self.node_list)),np.inf))
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
            if verbose=="on":
                print("\n"+"Iteration:"+str(iter1+1))
                print(dist_copy)
            iter1=iter1+1;
            

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
                    
    def writegraph(self,name="demofile.txt"):
        f = open(name, "w+")
        f.write(str(len(self.node_list))+" " + str(len(self.arclist))+"\n")
        for i in tqdm(range(len(self.arclist))):
            f.write(str(self.arclist[i][0].num)+" "+str(self.arclist[i][1].num)+" "+str(self.arclist[i][2])+" "+str(self.arclist[i][3])+"\n")
        f.close()
    
        


class node:
    i = 0

    def __init__(self, parent=[], child=[], cost=[], capacity=[]):
        self.num = node.i
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
        return self.__cost[self.__child.index(d)]

    def getcapacity(self, d):
        return self.__capacity[self.__child.index(d)]

    def addchild(self, child, cost=[], capacity=[]):
        if cost == []:
            cost = np.zeros(len(child))
        if capacity == []:
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



if __name__ == "main":
    gr = graph()
    gr.clear()
    gr.read_file("ExNet.txt")

    tmp = np.array(gr.getarclist())
    print(tmp[1, 1])
    # print(gr.getgraph())
    gr.dfs(gr.node_list[3])
    gr.get_arcinfo(0, 7)
