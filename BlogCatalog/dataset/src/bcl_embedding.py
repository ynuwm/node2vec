# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:38:13 2018
@author: ynuwm
"""
import networkx as nx
import numpy as np
from gensim.models import Word2Vec

G=nx.Graph()
GG=nx.Graph()

def readBCLData(filepath):  
    nodes_file = filepath + "nodes.csv"
    reader = open(nodes_file,encoding='utf-8').readlines()
    for line in reader:
        G.add_node(int(line[0]))

    edges_file = filepath + "edges.csv"
    reader = open(edges_file,encoding='utf-8').readlines()
    for line in reader:
        tmp = line.split(',')
        G.add_edge(int(tmp[0]),int(tmp[1]))
    del tmp

    G.to_undirected()
    print('data done')
    return

def alias_solve(prob):     
    small = list()
    large = list()
    length = len(prob)
    probtemp = np.zeros(length)
    aliasList = np.zeros(length, dtype=np.int)    
    
    # probability
    for k, x in enumerate(prob):
        probtemp[k] = prob[k] * length
        if probtemp[k] < 1:
            small.append(k)  # 小于1的下标
        else:
            large.append(k)

    while (len(small) > 0 and len(large) > 0):
        ss = small.pop()
        ll = large.pop()
        aliasList[ss] = ll
        probtemp[ll] -= 1 - probtemp[ss]
        if probtemp[ll] < 1:
            small.append(ll)
        else:
            large.append(ll)
    return aliasList, probtemp

alias=dict()

#from x to v
def getWeightsEdge(x,v,p,q):
    nbrs=sorted(G.neighbors(v))
    prob=list()
    for k in nbrs:
        if (k==x):
            prob.append(1/p)
        elif k in G.neighbors(x):  
            prob.append(1)
        else:
            prob.append(1/q)
    alias[(x,v)] = alias_solve(prob)
    return

    
def preprocessModifiedWeights(p,q):
    for edge in G.edges():
        getWeightsEdge(edge[0], edge[1],p,q)
        getWeightsEdge(edge[1], edge[0],p,q)
    return
 
    
def node2vecWalk(u,l): #从u出发走l步的list
    walk=[u]
    for i in range(l):
        curr=walk[-1]
        nbrs=sorted(G.neighbors(curr))
        if len(walk)==1:
            s=nbrs[sample(-1,curr)]
        else:
            s=nbrs[sample(walk[-2],curr)]
        walk.append(s)
    return walk


def sample(x,v):   
    V = sorted(G.neighbors(v))    
    if x==-1: #start node.
        return int(np.floor(np.random.rand()*len(V)))
    al,pro=alias[(x,v)]
    k=int(np.floor(np.random.rand()*len(al)))
    if np.random.rand()<pro[k]:
        return k
    else:
        return al[k]


def learnFeature(d,r,l,k,p,q):
    d = 128  # 维度
    r = 10   # 迭代次数
    l = 80   # 每次走多少步
    k = 10   # context 
    
    preprocessModifiedWeights(p,q)
    print("preprocess done")
    global walks
    walks=list()
    for i in range(r):
        for node in G.nodes():
            walk=node2vecWalk(node,l)
            walks.append(walk)
    print("walks done")
    learnEmmbeding(k,d,walks)
    print("emb done")
    return

    
np.save('../tmp/walks.npy',walks)    
np.save('../tmp/alias.npy',alias)  


def learnEmmbeding(k,d,walks):
    for i,line in enumerate(walks):
        for j,walk in enumerate(line):
            walks[i][j] = str(walk)
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1)
    model.save('../tmp/bcl_cbow_128_10_80_10_1_025.model')
    return



p = 1
q = 0.25

bclpathDir = '../BlogCatalog-dataset/data/'
readBCLData(bclpathDir)

learnFeature(128,10,80,10,p,q)


    
    
    
    
    
    
    
    
    
    
    
    
    
    