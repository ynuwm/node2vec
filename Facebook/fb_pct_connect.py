# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:44:10 2018
@author: ynuwm
"""
import copy
#import logging
import os
import pickle
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

# G=nx.Graph()
GG=nx.Graph()
remain_G=nx.Graph()
pct=0.5

def readFBData(filepath):
    filepath = '../facebook-dataset/facebook/'
 
    
    # global G
    global GG
    global remain_G
    pathDir = os.listdir(filepath)
    for efile in pathDir:
        if efile.endswith('.edges'):
            ego=int(efile.split('.')[0])
            # G.add_node(ego)
            GG.add_node(ego)
            # print ego
            child = os.path.join('%s%s' % (filepath, efile)) #文件路径
            print(child)
            # child='/Users/mac/Documents/gra/facebook/3980.edges'
            fopen = open(child, 'r')
            for line in fopen:
                x=int(line.split(' ')[0])
                y=int(line.split(' ')[1])
                # G.add_node(x)
                # G.add_node(y)
                GG.add_node(x)
                GG.add_node(y)
                GG.add_edge(x,y,samp=1)
                GG.add_edge(ego,x,samp=1)
                GG.add_edge(ego,y,samp=1)
                # if random.random() < pct:
                #     G.add_edge(x, y)
                #     GG.add_edge(x, y, samp=1)
                # else:
                #     GG.add_edge(x, y, samp=0)
                # if random.random() < pct:
                #     G.add_edge(ego, x)
                #     GG.add_edge(ego, x, samp=1)
                # else:
                #     GG.add_edge(ego, x, samp=0)
                # if random.random() < pct:
                #     G.add_edge(ego, y)
                #     GG.add_edge(ego, y, samp=1)
                # else:
                #     GG.add_edge(ego, y, samp=0)
            fopen.close()
    # random.shuffle(G.nodes())
    # random.shuffle(G.edges())
    GG.to_undirected()
    print("read done")
    remain_G=copy.deepcopy(GG)
    remain_G.to_undirected()
    print("copy done")

    
    edge_list=GG.edges(data=True)    
    edge_list = list(edge_list)
    samp_num=int(len(edge_list)*pct)
    
    shuffle_indices = np.random.permutation(np.arange(len(edge_list)))
     
    temp = list()
    for i in shuffle_indices:
        temp.append(edge_list[i])
    edge_list = temp
    del temp
    
    for i in range(samp_num):
        edge=edge_list[i]
        x=edge[0]
        y=edge[1]
        # z=edge[2]
        if GG[x][y]["samp"]==1:
            remain_G.remove_edge(x,y)
            if connected(remain_G):
                GG[x][y]["samp"]=0
            else:
                remain_G.add_edge(x,y,samp=1)
    del x,y,i
    
    f1 = open("graph_connect_2.txt", "wb")
    pickle.dump(GG, f1)
    f1.close()
    return  
    
def connected(g):
    len=0
    for c in nx.connected_components(g):
        len+=1
    if len>1:
        return False
    return True
    
    
    

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

#from t to v
def getWeightsEdge(x,v,p,q):
    nbrs=sorted(remain_G.neighbors(v))
    prob=list()
    for k in nbrs:
        if (k==x):
            prob.append(1/p)
        elif k in remain_G.neighbors(x):#x contects to t
            prob.append(1)
        else:
            prob.append(1/q)
    alias[(x,v)]=alias_solve(prob)
    return


def preprocessModifiedWeights(p,q):
    for edge in remain_G.edges():
        getWeightsEdge(edge[0], edge[1],p,q)
        getWeightsEdge(edge[1], edge[0],p,q)
    return

def node2vecWalk(u,l): #从u出发走l步的list
    walk=[u]
    for i in range(l):
        curr=walk[-1]
        nbrs=sorted(remain_G.neighbors(curr))
        if len(walk)==1:
            tmp=sample(-1, curr)
            # print(tmp)

            s=nbrs[tmp]
        else:
            s=nbrs[sample(walk[-2],curr)]
        walk.append(s)
    return walk

def sample(t,v):
    V = sorted(remain_G.neighbors(v))    
    if t==-1: #start node.
         
        return int(np.floor(np.random.rand()*len(V)))
    al,pro=alias[(t,v)]
    k=int(np.floor(np.random.rand()*len(al)))
    if np.random.rand()<pro[k]:
        return k
    else:
        return al[k]

def learnFeature(d,r,l,k,p,q):
    d = 128
    r = 10
    l = 80
    k = 10
    preprocessModifiedWeights(p,q)
    print("preprocess done")
    global walks
    walks=list()
    for i in range(r):
        for node in remain_G.nodes():
            nbrs=sorted(remain_G.neighbors(node))
            if len(nbrs)<=0:
                continue
            walk=node2vecWalk(node,l)
            walks.append(walk)
    print("walks done")
    learnEmmbeding(k,d,walks)
    return

def learnEmmbeding(k,d,walks):
    for i,line in enumerate(walks):
        for j,walk in enumerate(line):
            walks[i][j] = str(walk)
    #walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1)
    model.save('../tmp/fb_cbow_128_10_80_10_4_1_2.model')
    return

def loadData():
    fbpathDir = '../facebook-dataset/facebook/'
    readFBData(fbpathDir)
    print("load data done")
    return


p=4
q=1
loadData()
learnFeature(128,10,80,10,p,q)

