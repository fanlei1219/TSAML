import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from dataset_train import BasicDataset_train
import os
from PIL import Image


def load_data_cache(img_dir, mask_dir, supsz, querysz,class_con , class_n,task_num):
    dataset = BasicDataset_train(img_dir, mask_dir)
    img_spts = torch.FloatTensor(task_num, supsz, 3, 512, 512)
    mask_spts = torch.FloatTensor(task_num, supsz, 1, 512, 512)
    img_qrys = torch.FloatTensor(task_num, querysz, 3, 512, 512)
    mask_qrys = torch.FloatTensor(task_num, querysz, 1, 512, 512)

    train_idx=[]

    for n in range(task_num):
        img_spt = torch.FloatTensor(supsz, 3, 512, 512)
        mask_spt = torch.FloatTensor(supsz, 1, 512, 512)
        img_qry = torch.FloatTensor(querysz, 3, 512, 512)
        mask_qry = torch.FloatTensor(querysz, 1, 512, 512)

        task_n = np.where(class_con == class_n[n])[1]
        print (task_n)
        task_n.tolist()
        selected_datas_idx = random.sample(list(task_n), supsz + querysz)
        train_idx.extend(selected_datas_idx)

        selected_datas_spt_idx = selected_datas_idx[:supsz]
        selected_datas_qry_idx = selected_datas_idx[supsz:]

        for j, data in enumerate(selected_datas_spt_idx):
            img_spt[j] = dataset[data]['image']
            mask_spt[j] = dataset[data]['mask']

        for m, data in enumerate(selected_datas_qry_idx):
            img_qry[m] = dataset[data]['image']
            mask_qry[m] = dataset[data]['mask']

        img_spts[n] = img_spt
        mask_spts[n] = mask_spt
        img_qrys[n] = img_qry
        mask_qrys[n] = mask_qry

    return img_spts, mask_spts, img_qrys, mask_qrys,train_idx

def load_qn_meta(img_dir, mask_dir, querysz):

    dataset = BasicDataset_train(img_dir, mask_dir)
    img_qry = torch.FloatTensor(querysz, 3, 512, 512)
    mask_qry = torch.FloatTensor(querysz, 1, 512, 512)
    if len(dataset)>querysz:
        selected_datas_idx = random.sample(range(len(dataset)), querysz)
    else:
        selected_datas_idx = range(len(dataset))



    #print (selected_datas_idx)

    for i, data in enumerate(selected_datas_idx):
        img_qry[i] = dataset[data]['image']
        mask_qry[i] = dataset[data]['mask']

    return img_qry, mask_qry



def data_load_all(img_dir):
    all_imgs = os.listdir(img_dir)
    num_img = len(all_imgs)
    all_img = torch.FloatTensor(num_img, 3, 512, 512)
    i = 0
    for test_path in all_imgs:
        idx = test_path.split('.')[0]
        img = Image.open(img_dir + idx +'.png')
        img = np.array(img)
        tot = transforms.ToTensor()
        img = tot(img)
        all_img[i] = img
        i = i + 1
    return all_img





















