#-*-coding:utf-8-*-
import sys
import os
import numpy as np
import random
#from tqdm import tqdm
# from setting import *
from copy import deepcopy

def tqdm(a):
    try:
        from tqdm import tqdm
        return tqdm(a)
    except:
        return a

trainset = []       # for a dataset, 
trainset.append([]) # train for trainset[0]
trainset.append([]) # dev for trainset[1]
trainset.append([]) # test for trainset[2]

is_train = False

data_size = 3000
var_len = 80
classnum = 5
embedding_size = 128
batch_size = 12800

def edge2d(edge):
    mat = np.zeros([var_len, var_len])
    for i in range(len(edge)):
        mat[i, i] = len(edge[i])
    return mat

def edge2mat_update(edge, maps):
    mat = np.zeros([var_len, var_len])
    for i in range(min(len(maps), len(edge))):
        for t in range(min(len(maps), len(edge))):
            mat[maps[i], maps[t]] = edge[i, t]

    return mat



def edge2mat(edge):
    mat = np.zeros([var_len, var_len])
    for i in range(len(edge)):
        l = edge[i]
        for key in l:
            mat[i, key] = 1

    return mat

def read_data (file_name):
    file2number = {}
    file2number["train.txt"] = 0
    file2number["dev.txt"] = 1
    file2number["test.txt"] = 2

    index_of_dataset = file2number[file_name]
    f = open(file_name, "r")
    file_data = []
    for i in range(3):
        file_data.append([])

    lines = f.readlines()
    all_vec = []
    count = 0 
    each_vec = []

    bf = ""
    rules_line = ""
    father = []
    now_site = 0
    perm = None
    for i in tqdm(range(len(lines))):
        if index_of_dataset == 0 and len(all_vec) >= data_size:
            break
        elif len(all_vec) >= 50000:
                break
        lines[i] = str(lines[i]).strip()
        t = i % 3
        if t == 2 : 
            each_vec.append(int(lines[i].strip()))
            if each_vec[-1] != 4 and each_vec[0] > 40:
                all_vec.append(deepcopy(each_vec))
            each_vec = []
        elif t == 0:
            each_vec.append(int(lines[i].strip()))
            perm = np.random.permutation(int(lines[i].strip()))
            maps = np.zeros(var_len)
            for k in range(min(len(perm), len(maps))):
                maps[k] = perm[k] + 1
            each_vec.append(maps)
            mask = np.zeros([var_len])
            for t in range(int(lines[i].strip())):
                mask[t] = 1
            each_vec.append(mask)
        else:
            each_vec.append(edge2d(eval(lines[i])))
            each_vec.append(edge2mat(eval(lines[i])))

    trainset[index_of_dataset] = deepcopy(all_vec)



def random_data (dataset): # shuffle training set  
    random.shuffle(dataset)

def batch_data (batch_size, dataset_name): # get an acceptable data for NN;
    dic = {}
    dic["train"] = 0
    dic["dev"] = 1
    dic["test"] = 2
    index_of_dataset = dic[dataset_name]

    global trainset
    data = trainset[index_of_dataset]
    all_data = []
    all_index = []
    data_now = data

    if index_of_dataset == 0: # random training data;
        random.shuffle(data_now)
    
    all_data = [data_now[site: min(len(data_now), site + batch_size)] for site in range(0, len(data_now), batch_size)]
    ret_data = []
    # batch/ batch-size / 3 / data
    for site in tqdm(range(len(all_data))):
        t = 1
        if index_of_dataset == 0: # train model
            t = 1#30
        for i in range(t):
            now = all_data[site]
            each_data = []
            perm = np.random.permutation(now[i][0])
            for key in range(6):
                if key == 1:
                    for i in range(len(now)):
                        now[i][key] = np.zeros([var_len])
                        for q in range(min(len(now[i][key]), len(perm))):
                            now[i][key][q] = perm[q] + 1

                each_data.append([now[i][key] for i in range(len(now))])
            ret_data.append(each_data)
    
    return ret_data, ret_data

def resolve_data():
    global trainset
    read_data("train.txt")
    read_data("dev.txt")
    read_data("test.txt")

resolve_data()
