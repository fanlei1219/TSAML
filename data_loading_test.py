import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from dataset import BasicDataset
from dataset_train import BasicDataset_train
import os


def load_data_cache2(img_dir_train, mask_dir_train, img_dir_test, mask_dir_test, sup_idx, k_spt,  batch_size):
    dataset_train = BasicDataset_train(img_dir_train, mask_dir_train)
    dataset_test = BasicDataset(img_dir_test, mask_dir_test)
    setsz = k_spt
    querysz = 1

    img_spts = torch.FloatTensor(batch_size, setsz, 3, 512, 512)
    mask_spts = torch.FloatTensor(batch_size, setsz, 1, 512, 512)
    img_qrys = torch.FloatTensor(batch_size, querysz, 3, 512, 512)
    mask_qrys = torch.FloatTensor(batch_size, querysz, 1, 512, 512)

    #if not os.path.exists('./test_idx'):
     #   os.mkdir('./test_idx')

    #txt1 = './test_idx/spt_ids.txt'
    #txt2 = './test_idx/qry_ids.txt'
    #f1 = open(txt1, 'w+')
    #f2 = open(txt2, 'w+')
    for i in range(batch_size):
        img_spt = torch.FloatTensor(setsz, 3, 512, 512)
        mask_spt = torch.FloatTensor(setsz, 1, 512, 512)
        img_qry = torch.FloatTensor(querysz, 3, 512, 512)
        mask_qry = torch.FloatTensor(querysz, 1, 512, 512)

        #selected_datas_idx = random.sample(range(len(dataset_test)), setsz+querysz)###测试

        #selected_datas_spt_idx = random.sample(range(len(dataset_train)), setsz)
        selected_datas_spt_idx = sup_idx[i,:]

        #print(selected_datas_spt_idx )
        selected_que_idx = range(len(dataset_test))
        #selected_datas_qry_idx = random.sample(range(len(dataset_test)), querysz)
        selected_datas_qry_idx = selected_que_idx[i*querysz:(i+1)*querysz]
        #print(selected_datas_qry_idx)

        #selected_datas_spt_idx = selected_datas_idx[:setsz]
        #selected_datas_qry_idx = selected_datas_idx[setsz:]
        # img_spt, mask_spt, img_qry, mask_qry = [], [], [], []
        # selected_datas_spt = np.random.choice(dataset, setsz, replace=True)
        # selected_datas_query = np.random.choice(dataset, querysz, replace=True)
        for j, data in enumerate(selected_datas_spt_idx):
            img_spt[j] = dataset_train[data]['image']##j是batch里面第几个，data是数据里面第几个
            mask_spt[j] = dataset_train[data]['mask']
            context1 = str(i) + '\t' + str(data) + '\n'
            #f1.write(context1)
        for m, data in enumerate(selected_datas_qry_idx):
            img_qry[m] = dataset_test[data]['image']
            mask_qry[m] = dataset_test[data]['mask']
            context2 = str(i) + '\t' + str(data) + '\n'
            #f2.write(context2)

        img_spts[i] = img_spt
        mask_spts[i] = mask_spt
        img_qrys[i] = img_qry
        mask_qrys[i] = mask_qry


    return img_spts, mask_spts, img_qrys, mask_qrys





































