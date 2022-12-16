import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
from torch.utils.data import DataLoader
import torch
from PIL import Image
from unet import UNeT
import os
import torch.nn as nn
from kneed import KneeLocator



def get_dis_con(feature_con,num_img):
    var_con = torch.var(feature_con, 1, keepdim=True)
    ave_con = torch.mean(feature_con, 1, keepdim=True)
    feature_standard = (feature_con - ave_con) / (var_con)
    feature_standard = torch.where(torch.isnan(feature_standard), torch.full_like(feature_standard, 0), feature_standard)
    dis_con = torch.empty(num_img,num_img)
    for i in range(num_img):##样本i
        for j in range(i):
            sim = torch.sum((feature_standard[:,i]-feature_standard[:,j]) **2)
            dis_con[i][j]= torch.sqrt(sim)
            dis_con[j][i] = torch.sqrt(sim)
        dis_con[i][i] = 0
    return dis_con

def ja_sim_con(dis_con, k, num_img):
    d_con, _ = torch.sort(dis_con, dim=1)
    ja_con = torch.empty(num_img, num_img)
    p_con = torch.empty(num_img, num_img)
    for m in range(num_img):
        p_con[m,:] = torch.where(dis_con[m,:]<= d_con[m][k], 1 , 0)

    we_ja_con = p_con * dis_con
    for i in range(num_img):  ##样本i
        for j in range(i):
            min_n = 0
            max_n = 0
            for n in range(num_img):
                min_n = min_n + torch.min(we_ja_con[i][n],we_ja_con[j][n])
                max_n = max_n + torch.max(we_ja_con[i][n],we_ja_con[j][n])

            ja_con[i][j] = 1-min_n / max_n
            ja_con[j][i] = 1-min_n / max_n
        ja_con[i][i] = 0
    return ja_con

def get_eps(num_img,ja_con, k):
    sorted, _ = torch.sort(ja_con, dim=1)
    dis_k = torch.mean(sorted[:,:k+1],1)
    sorted1,_ = torch.sort(dis_k, dim=0, descending=False)
    x = [i for i in range(num_img)]
    y = sorted1.tolist()
    kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='increasing', online=False)
    #plt.plot(x, y)
    #plt.show()
    eps = kneedle.knee_y
    print(eps)
    return eps

def find_neighbor(num_img, Dis, eps):#x为文件夹
    N = []
    for j in range(num_img):
        temp = Dis [j]
        if temp <= eps:
            N.append(j)
    N = list(set(N))
    return N


def get_clu_task(ja_con, min_num, eps ,num_img):
    sub_class = np.zeros((1, num_img),dtype=np.int16)
    ja_con = ja_con.numpy()
    core =[]
    for i in range(num_img):
        D = ja_con[i,]
        X = find_neighbor(num_img, D, eps)  # 图像下标
        #print(X)
        if len(X) == 1:
            sub_class[0, i] = -1  # 无类别
        if len(X) >= min_num +1:
            sub_class[0,i] = i+1
            core.append(i)
            for x in X:

                if sub_class[0, x] == 0:
                    sub_class[0, x] = i+1

    n = 0
    for i in core:#核心点 i
        n = n+1
        b_core = core[n:]
        for j in b_core:
            if ja_con[i,j]<eps:
                sub_class = np.where(sub_class == sub_class[0,j], sub_class[0,i],sub_class)

    X_2 = ((sub_class == 0).nonzero())[1]
    for x in X_2:
        sub_class[0, x] = -1

    return sub_class,core


def dbscan_task(num_img, feature_con):
    sim_con = get_dis_con(feature_con, num_img)
    ja_con = ja_sim_con(sim_con, 10, num_img)
    ##eps = get_eps(num_img, ja_con, 10)
    sorted, _ = torch.sort(ja_con, dim=1)
    sorted1, _ = torch.sort(sorted[:, 10], dim=0, descending=False)
    eps = sorted1[18]
    print(eps)

    class_con, core = get_clu_task(ja_con, 7, eps, num_img)
    return class_con,core


def test_sup(n_s,n_q,feature_s,feature_q):
    feature_con = np.append(feature_q, feature_s,axis=1)
    var_con = torch.var(feature_con, 1, keepdim=True)
    ave_con = torch.mean(feature_con, 1, keepdim=True)
    feature_standard = (feature_con - ave_con) / (var_con)
    feature_standard = torch.where(torch.isnan(feature_standard), torch.full_like(feature_standard, 0),
                                   feature_standard)
    feature_q = feature_standard[:,:2]
    feature_s = feature_standard[2:,:]
    dis_con = torch.empty(n_q, n_s)
    for i in range(n_q):
        for j in range(n_s):
            sim = torch.sum((feature_q[:, i] - feature_s[:, j]) ** 2)
            dis_con[i][j] = torch.sqrt(sim)
    return dis_con


























