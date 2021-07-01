import numpy as np
import networkx as nx
from tqdm import tqdm

class graph:

    def __init__(self):
        node.i = 0
        self.node_list = []
        self.graph = {}
        self.arclist = []
        self.path_dfs = []
        self.parent_list = []

    def create_nodes(self, m):
        tmp = [node() for i in range(m)]
        self.node_list = self.node_list  +tmp
        self.parent_list = self.node_list.copy()
        return tmp

    def read_file(self,name):
        file1 = open(name, "r+")
        file1.seek(0)
        a = file1.readlines()
        for i in range(len(a)):
            a[i] = a[i].replace("\n", "")
            a[i] = a[i].split(" ")
            a[i] = list(map(int, a[i]))

        self.add_arcs(a)
        return a

    def add_arcs(self, n):
        if isinstance(n, dict):
            for a in n.keys():
                a.addchild(n[a])
        else:
            for i in tqdm(range(len(n)),desc="Loading File"):
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

    def draw(self, d=None):

        if d == None:
            d = self.getgraph()

        g = nx.DiGraph()
        g.add_nodes_from(d.keys())
        for k, v in d.items():
            g.add_edges_from(([(k, t) for t in v]))

        nx.draw(g, with_labels=True)

    def dfs(self, s, e=None):
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e]
        visited = []
        path = []
        visited.append(s)
        path.append(s)
        self.path_dfs = []

        self.dfs_algo(visited, path, s, e)
        if set(visited) == set(self.node_list):
            print("All nodes can be visited")
        else:
            print("Nodes that haven't be visited:" + str(set(self.node_list) - set(visited)))
        return (visited, self.path_dfs)

    def dfs_algo(self, visited, path, s, e):

        for i in s.getchild():
            if i not in visited:
                visited.append(i)
                if e == i:

                    self.path_dfs = (path + [i])
                    break
                else:
                    try:
                        self.dfs_algo(visited, path + [i], i, e)
                    except:
                        print(i)
                        print(self.dfs_algo(visited, path, i, e))
                        
    def search(self,s,e=None,method = "bfs"):
        if isinstance(s,int):
            s = self.node_list[s]
        if isinstance(e,int):
            e = self.node_list[e]
        
        qu = [s];
        unexpl = self.node_list.copy()
        
        visited = [];
       
        
        while not (unexpl ==[] or qu==[]): 
 
            s = qu[0]
            visited.append(s)
            
            if s == e:
                prt = e
                path = []
                while 1:
                    if prt==visited[0]:
                        return [prt]+path
                    else:
                        path =[prt]+path
                        #ind = self.node_list.index(prt)
                        prt = prt.temp_parent
                   
                    
            
            #tmp = [a for a in s.getchild() if a not in visited and a not in qu]
            tmp   = list(set(s.getchild()) - set(visited) - set(qu))
            for i in tmp:
                i.temp_parent = s
            #self.parent_list = np.array(self.parent_list)
            #self.parent_list[tmp] = s
            
            if method == "bfs":
                qu =  qu + tmp
               
            else:
                qu = tmp + qu
               
            qu.remove(s)
            #print(s,unexpl,qu)
            unexpl.remove(s)
           
        #print(self.node_list)  
        return (visited, unexpl)
        
    def mod_bell_ford(self,s,e):
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
            print("\n"+"Iteration:"+str(iter1))
            for i in self.node_list:
                print(i.dist,end=" ")
            iter1=iter1+1;
                    
        if e.dist==np.inf:
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
