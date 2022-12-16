import torch
from torchvision import transforms
import os
import numpy as np
import cv2
from data_loading import load_data_cache
from data_loading import data_load_all
from data_loading import load_qn_meta
from torch.utils.data import DataLoader
import random
import argparse
from maml import MetaLearner
from unet import *
import torch.utils.data as Data
from PIL import Image
from data_loading_test import load_data_cache2
from collections import Counter


def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maml = MetaLearner(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    img_dir_train = './data/train/image/'
    mask_dir_train = './data/train/label/'

    img_sup_test1 = './data/test/sup/image/'
    mask_sup_test1 = './data/test/sup/label/'
    img_que_test1 = './data/test/que/image/'
    mask_que_test1 = './data/test/que/label/'

    meta_batch = 300
    task_num = 2
    img_all = data_load_all(img_dir_train)
    img_all = img_all.to(device)
    test_sup_all = data_load_all(img_sup_test1)
    test_sup_all = test_sup_all.to(device)
    test_que_all = data_load_all(img_que_test1)
    test_que_all = test_que_all.to(device)


    miou = []
    loss_q = []
    txt = './class.txt'
    f = open(txt, 'w+')
    for k in range(meta_batch):
        save_P = './test_save/' + str(k) + '/'
        if not os.path.exists(save_P):
            os.makedirs(save_P)

        class_con, core = maml.get_task(img_all, k)

        context = str(k) + '\t' + str(class_con) + '\t'+str(core) +'\n'
        f.write(context)


        c_n = []
        class_new = np.delete(class_con, np.where(class_con == -1))
        cnum = Counter(class_new.tolist())
        for key, value in cnum.items():
            if value > 6:
                c_n.append(key)
        if len(c_n) > 1:
            class_n = random.sample(c_n, task_num)
        else:
            class_n = c_n
            class_n.append(c_n[0])
        print(class_n)

        img_spts, mask_spts, img_qrys, mask_qrys, train_idx= load_data_cache(img_dir_train, mask_dir_train,
                                                                               args.k_spt, args.k_qry, class_con, class_n, task_num)

        qn_allimg = list(set(core) - set(train_idx))

        i = 0
        img_dir = os.listdir(img_dir_train)
        for img_path in img_dir:
            img_name = img_path.split('.')[0]
            img = Image.open(img_dir_train + img_name + '.png')
            if (i in qn_allimg):
                core_path = './test_save/' + str(k) + '/core/'
                if not os.path.exists(core_path):
                    os.makedirs(core_path)
                core_save = core_path + img_name + '.png'
                img.save(core_save)
            i = i + 1

        qn_path = './test_save/' + str(k) + '/core/'
        qn_img, qn_mask = load_qn_meta(qn_path, mask_dir_train, 2 * args.k_qry)
        img_spts, mask_spts, img_qrys, mask_qrys = img_spts.to(device), mask_spts.to(device), img_qrys.to(device), mask_qrys.to(device)
        qn_img, qn_mask = qn_img.to(device), qn_mask.to(device)
        iou, loss = maml(img_spts, mask_spts, img_qrys, mask_qrys, qn_img, qn_mask)
        miou.append(iou)
        loss_q.append(loss)
        print(print('step:', k, '.\ttraining iou:', iou, '\ttraining loss:', loss))

        if (k + 1) % 5 == 0:
            sup_idx = maml.get_sup(test_sup_all, test_que_all)
            sup_idx = sup_idx[:,:args.k_spt]
            batch_size_test = 300
            img_spts1, mask_spts1, img_qrys1, mask_qrys1 = load_data_cache2(img_sup_test1, mask_sup_test1,img_que_test1,mask_que_test1,
                                                                            sup_idx,args.k_spt, batch_size_test)

            dataset_test1 = Data.TensorDataset(img_spts1, mask_spts1, img_qrys1, mask_qrys1)
            db_test1 = Data.DataLoader(dataset_test1, 1, shuffle=False, num_workers=0, pin_memory=True,
                                       sampler=torch.utils.data.sampler.SequentialSampler(dataset_test1))

            for it, (img_spt, mask_spt, img_qry, mask_qry) in enumerate(db_test1):
                img_spt, mask_spt, img_qry, mask_qry = img_spt.squeeze(0).to(device), mask_spt.squeeze(0).to(
                    device), img_qry.squeeze(0).to(device), mask_qry.squeeze(0).to(device)


                if not os.path.exists('./test_label/' + str(k) + '/'):
                    os.makedirs('./test_label/' + str(k) + '/')
                if not os.path.exists('./test_pred_masks/' + str(k) + '/'):
                    os.makedirs('./test_pred_masks/' + str(k) + '/')
                iou_test = 0
                mask_pred, iou_t = maml.finetunning(img_spt, mask_spt, img_qry, mask_qry)
                # print(mask_pred.size())
                iou_test = iou_t + iou_test

                for m in range(1):
                    result = (mask_pred[m].cpu().numpy()).squeeze(0)
                    result = Image.fromarray((result * 255).astype(np.uint8))
                    result_path = './test_pred_masks' + '/' + str(k) + '/' + '{}_{}_{}.png'.format(1, it, m)  ##预测
                    result.save(result_path)

                    img = (mask_qry[m].cpu().numpy()).squeeze(0)
                    img = np.where(img > 0, 255, 0)
                    img = Image.fromarray((img).astype(np.uint8))
                    img_path = './test_label' + '/' + str(k) + '/' + '{}_{}_{}.png'.format(1, it, m)  ##mask
                    img.save(img_path)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', type=int, help='epoch number', default=1)
    # parser.add_argument('--n_way', type=int, help='n way', default=1)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    # parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)####batch_size
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update lerning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)

    args = parser.parse_args()

    main()




