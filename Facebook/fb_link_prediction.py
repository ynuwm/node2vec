# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:44:10 2018
@author: ynuwm
"""
import pickle
import random
import gensim

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import linear_model
from sklearn import metrics

biop=1
biop_str={1:"average",2:"hadamard",22:"hadamard2",3:"weightedl1",4:"weightedl2"}
train_data_num=15000
logreg=""
lin_svc=""
tt=False

def load_data():
    global G
    f2 = open("./graph_connect_2.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()
    G = load_list

    model_path ='../tmp/fb_cbow_128_10_80_10_4_1_2.model'
    print(model_path)
    global model
    model = gensim.models.Word2Vec.load(model_path)
    # print G.edges(data=True)
    global nodes_num
    nodes_num=len(G.nodes())
    print("load data done")

    nbs=G.neighbors(1233)
    for n in nbs:
        print(G[1233][n])
        

def test():
    global logreg
    global train_negative_list
    global biop
    test_len=10000
    actual=list()
    pred=list()
    ve=list()

    print(logreg.get_params())

    for edge in G.edges(data=True):
        if edge[2]['samp']==0:
            if random.random()>(test_len+0.0)/(40000-train_data_num):
                continue
            vec=[solve(edge[0],edge[1],biop)]

            prob=logreg.predict_proba(vec)#[1]   
            # prob=logreg.decision_function(vec)
            # print vec
            if tt == True:
                print(prob)
            actual.append(1)
            pred.append(prob[0][1])
            ve.extend(logreg.predict(vec))
    if tt == True:
        print(actual)
        print(ve)
        print(pred)
    print("test positive ",len(actual))
    cnt=0
    node_list=list(G.nodes(data=True))
    while True:
        node1 = node_list[random.randrange(nodes_num)][0]
        node2 = node_list[random.randrange(nodes_num)][0]
        V = sorted(G.neighbors(node1))    
        if not (node2 in V):
            if (node1,node2) in train_negative_list:  
                continue
            vec=[solve(node1, node2, biop)]
            prob = logreg.predict_proba(vec)
            
            if tt == True:
                print(prob)
            cnt += 1
            actual.append(0)
            pred.append(prob[0][1])
            ve.extend(logreg.predict(vec))
        if cnt >= test_len:
            break
    print("test negative ",cnt)

    print(actual)
    print(ve)
    print(pred)

    print("prepare test done")
    roc_auc = metrics.roc_auc_score(actual, pred)
    print(roc_auc)
    print(biop_str[biop])
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.title(str(biop_str[biop])+" auc:"+str(roc_auc))
    plt.plot(fpr, tpr)
    plt.show()
    return



def prepare_train():
    global train_positive_list
    global train_negative_list
    global nodes_num
    train_positive_list = list()
    train_negative_list=list()
    pl = 0
    
    #random.shuffle(G.edges(data=True))   
    edge_list=list(G.edges(data=True))    
    shuffle_indices = np.random.permutation(np.arange(len(edge_list)))
     
    temp = list()
    for i in shuffle_indices:
        temp.append(edge_list[i])
    edge_list = temp
    del temp
    
    for edge in edge_list:
        if edge[2]['samp']==1:
            if random.random() > 0.2:
                pl += 1
                x=edge[0]
                y=edge[1]
                train_positive_list.append((x,y))
                if pl>=train_data_num:
                    break

    cnt=0
    node_list=list(G.nodes(data=True))
    while True:
        node1 = node_list[random.randrange(nodes_num)][0]
        node2 = node_list[random.randrange(nodes_num)][0]
        if node1==node2:
            continue
        V = sorted(G.neighbors(node1))    
        if not (node2 in V):
            train_negative_list.append((node1, node2))
            cnt+=1
        if cnt>=train_data_num:
            break

        
    print('train positive',pl,' , negative ',cnt)

    nX = train_positive_list[:]
    nX.extend(train_negative_list)
    # random.shuffle(nX)
    vX = [solve(edge[0],edge[1],biop) for edge in nX]
    Y = [0 if edge in train_negative_list else 1 for edge in nX]
    # print Y
    global logreg
    logreg = linear_model.LogisticRegression()
    logreg.fit(vX, Y)
    # logreg.
    if tt == True:
        for v in vX:
            print(v)


    global lin_svc
    lin_svc = svm.LinearSVC()
    lin_svc.fit(vX, Y)

    print("train lr done")

    
def solve(x,y,para):
    if para==1:
        return average(x,y)
    if para==2:
        return hadamard(x,y)
    if para==22:
        return hadamard_2(x,y)
    if para==3:
        return weightedl1(x,y)
    if para==4:
        return weightedl2(x,y)

def average(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [(a+b+0.0)/2 for a, b in zip(vec1, vec2)]
    return prod

def hadamard(x,y):
    global model
    # for n in G.neighbors(x):
    #     if G[x][n]['samp']==1:
    #         print 'okx'
    #         break
    # for n in G.neighbors(y):
    #     if G[y][n]['samp'] == 1:
    #         print 'oky'
    #         break
    # print G.neighbors(y)
    vec1=model[str(x)]
    vec2=model[str(y)]
    prod = [a * b for a, b in zip(vec1, vec2)]
    return prod

def hadamard_2(x,y):
    global model
    vec1=model[str(x)]
    tmp1=sum([a*a for a in vec1])**0.5
    vec2=model[str(y)]
    tmp2 = sum([a * a for a in vec2]) ** 0.5
    prod = [(a * b)/(tmp1*tmp2) for a, b in zip(vec1, vec2)]
    return prod

def weightedl1(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [abs(a - b) for a, b in zip(vec1, vec2)]
    return prod

def weightedl2(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [(a - b)*(a - b) for a, b in zip(vec1, vec2)]
    return prod


load_data()
prepare_train()
test()

