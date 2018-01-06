# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:38:13 2018
@author: ynuwm
"""
import copy
import random

import gensim
import sklearn.svm as svm

train_data = list()
test_data = list()
pct = 0.8
nodes_groups_pred=dict()

clf_model=dict()
group_list=list()
nodes_groups=dict()

model_path='../tmp/bcl_cbow_128_10_80_10_1_025.model'

def prepare_data_for_train():
    for node in nodes_groups:
        if random.random()<pct:
            train_data.append(node)
        else:
            test_data.append(node)
    print(len(train_data),len(test_data))

def load_data():
    global model
    model = gensim.models.Word2Vec.load(model_path)
    print(len(model['2']))
    
    
    group_list = list()
    group_path = '../BlogCatalog-dataset/data/groups.csv'
    reader = open(group_path,encoding='utf-8').readlines()
    
    for line in reader:
        group_list.append(int(line))
    global group_list
        
    
    nodes_groups = {}     
    group_edges_path = '../BlogCatalog-dataset/data/group-edges.csv'
    reader = open(group_edges_path,encoding='utf-8').readlines()
    
    for line in reader:
        line = line.split(',')
        node=int(line[0])
        group=int(line[1])
        if node in nodes_groups.keys():
            nodes_groups[node].append(group)  # node has label i
        else:
            nodes_groups[node] = list()
            nodes_groups[node].append(group)
    global nodes_groups
       
    
def prepare_classification(target):
    positive_list=list()
    negative_list=list()

    for node, llist in nodes_groups.items():
        if (node in train_data):
            if (target in llist):
                positive_list.append(node)
            else:
                negative_list.append(node)
    positive_list.extend(positive_list)
    tmp=positive_list[:]
    positive_list.extend(positive_list)
    positive_list.extend(tmp)

    print(len(positive_list),len(negative_list))
    
    positive_list.extend(negative_list)
    random.shuffle(positive_list)
    nX=positive_list[:]
    vX=[model[str(node)] for node in nX]
    Y=[0 if node in negative_list else 1 for node in nX]
    # print Y
    lin_svc=svm.LinearSVC()
    lin_svc.fit(vX, Y)
    clf_model[target]=copy.deepcopy(lin_svc)


def test():
    print(model_path)
    tp=dict.fromkeys(group_list,0)
    pre_total=dict.fromkeys(group_list,0)
    tru_total=dict.fromkeys(group_list,0)
    for node in test_data:
        global nodes_groups_pred
        nodes_groups_pred[node]=[]
        for i in group_list:
            pre_label=clf_model[i].predict([model[str(node)]])
            if (pre_label==1):
                # print "pre"
                nodes_groups_pred[node].append(i)


    for node in test_data:
        pred=nodes_groups_pred[node]
        truth=nodes_groups[node]
        print("pred:  ",pred)
        print("truth:",truth)
        for p_label in pred:
            if (p_label in truth):
                tp[p_label]+=1
            pre_total[p_label]+=1

    for k,list in nodes_groups.items():
        if k in test_data:
            for label in list:
                if label in tru_total.keys():
                    tru_total[label]+=1
                else:
                    tru_total[label]=0

    total_label=0
    f1=dict.fromkeys(group_list,0)
    for label in group_list:
        if pre_total[label]+tru_total[label]==0:
            continue
        if (tru_total[label]>0):
        # if tp[label] > 0:
            total_label+=1
        f1[label]=(2*tp[label]+0.0)/(pre_total[label]+tru_total[label])
    print("num of nodes:",len(nodes_groups.keys()))
    print(f1)
    print(tp)
    print(pre_total)
    print(tru_total)

    tmp=[]
    for k,v in tp.items():
        if v<=0:
            tmp.append(k)
    print(tmp)
    print(total_label)
    print("macro-f1:",sum(f1.values())/total_label)
    print("micro-f1:",2*(sum(tp.values())+0.0)/(sum(pre_total.values())+sum(tru_total.values())))
    print(model_path)
    print("svm")


load_data()
prepare_data_for_train()
for label in group_list:
    prepare_classification(label)








