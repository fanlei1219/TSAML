import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
from IoU import iou
from unet import UNeT
import torch.nn as nn
from loss import criterion
from loss import metric
from task_clu import dbscan_task
from task_clu import ja_sim_con
from task_clu import get_dis_con
from PIL import Image

class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.net = UNeT()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, img_spt, mask_spt, img_qry, mask_qry, qn_img, qn_mask):#一个meta_batch
        task_num, img_num, channels, h, w = img_spt.size()
        loss_list_qry = [0 for _ in range(task_num)]
        iou_list_qry = [0 for _ in range(task_num)]
        loss_list_meta = [0 for _ in range(task_num)]


        for i in range(task_num):#1个task的数据

            mask_hat,_ = self.net(img_spt[i], params=None, bn_training=True)#初始化参数预测
            loss = criterion(mask_hat, mask_spt[i])#更新前loss
            #print (loss)

            grad = torch.autograd.grad(loss, self.net.parameters())

            tuples = zip(grad, self.net.parameters())#得到梯度与参数的数组
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], tuples))#内循环梯度下降（该任务参数）


            for k in range(1, self.update_step):####1234
                mask_hat,_ = self.net(img_spt[i], fast_weights, bn_training=True)
                loss = criterion(mask_hat, mask_spt[i])
                #print(loss)

                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], tuples))###内循环更新参数

                if k == self.update_step - 1:#
                    mask_hat,_ = self.net(img_qry[i], fast_weights, bn_training=True)
                    qn_hat,_ = self.net(qn_img, fast_weights, bn_training=True)

                    #for mn in range (4):
                     #   probs = torch.sigmoid(qn_hat[mn])
                       # probs = torch.where(probs> 0.5, 1, 0)
                        #result = (probs.cpu().numpy()).squeeze(0)

                        #target = (qn_mask[mn].cpu().numpy()).squeeze(0)
                        #target = Image.fromarray((target * 255).astype(np.uint8))
                        #result = Image.fromarray((result * 255).astype(np.uint8))
                        #result_path = './qn_pred' + '/' +str(i)+ '/' +'{}_{}.png'.format(1,mn)  ##预测
                        #target_path = './qn_pred' + '/' + str(i) + '/' +'{}_{}.png'.format(2,mn)
                        #result.save(result_path)
                        #target.save(target_path)


                    loss_list_qry[i] = criterion(mask_hat, mask_qry[i])



                    with torch.no_grad():
                        loss_list_meta[i] = metric(qn_hat, qn_mask)
                        pred_qry = torch.sigmoid(mask_hat)
                        pred_qry = torch.where(pred_qry > 0.5, 1, 0)
                        iou_each = iou(mask_qry[i],pred_qry).item()
                        iou_list_qry[i] += iou_each



        with torch.no_grad():
            m_iou = (np.array(iou_list_qry)).sum() / (task_num)
            loss = (np.array(loss_list_qry)).sum() / (task_num)

        print(loss_list_meta)

        m_qn = list(map(lambda num: num**2, loss_list_meta))
        #m_qn = loss_list_meta
        loss_all = 0
        for n in range(task_num):
            loss_all = loss_all + m_qn[n]
        loss_qry = 0
        for n in range(task_num):
            wei = m_qn[n]/ loss_all
            loss_qry = loss_list_qry[n]* wei + loss_qry
            print(wei)


        self.meta_optim.zero_grad()
        loss_qry.backward()####外循环meta_lr更新初始参数
        self.meta_optim.step()
        #torch.save(self.net.state_dict(), "./checkpoints/")
        return m_iou, loss

    def get_task(self, img_all, meta_step):
        net = deepcopy(self.net)
        num_img, channels, h, w = img_all.size()
        feature_con = torch.empty(512, num_img)
        if meta_step == 0:
            with torch.no_grad():
                for i in range(num_img):
                    img_input = img_all[i].unsqueeze(0)
                    _,low_feature = net(img_input, params=None, bn_training=True)
                    feature_i = nn.functional.adaptive_avg_pool2d(low_feature, output_size=(1, 1))
                    feature = np.squeeze(feature_i)
                    feature_con[:, i] = feature

        else:
            with torch.no_grad():
                for i in range(num_img):
                    img_input = img_all[i].unsqueeze(0)
                    _, low_feature = net(img_input, net.parameters(), bn_training=True)
                    feature_i = nn.functional.adaptive_avg_pool2d(low_feature, output_size=(1, 1))
                    feature = np.squeeze(feature_i)
                    feature_con[:, i] = feature
        print (feature_con.shape)

        class_con,core = dbscan_task(num_img ,feature_con)

        del net
        return class_con,core

    def get_sup(self, img_sup,img_que):
        net = deepcopy(self.net)
        n_s, channels, h, w = img_sup.size()
        n_q,_,_,_= img_que.size()
        num_img = n_s + n_q
        feature_con= torch.empty(512, n_s + n_q)

        with torch.no_grad():
            for i in range(n_s):
                img_input = img_sup[i].unsqueeze(0)
                _, low_feature = net(img_input, net.parameters(), bn_training=True)
                feature_i = nn.functional.adaptive_avg_pool2d(low_feature, output_size=(1, 1))
                feature = np.squeeze(feature_i)
                feature_con[:, i] = feature
            for j in range(n_q):
                img_input = img_que[j].unsqueeze(0)
                _, low_feature = net(img_input, net.parameters(), bn_training=True)
                feature_j = nn.functional.adaptive_avg_pool2d(low_feature, output_size=(1, 1))
                feature = np.squeeze(feature_j)
                feature_con[:, n_s + j] = feature

        dis_con = get_dis_con(feature_con, num_img)
        ja_con = ja_sim_con(dis_con, 10, num_img)
        ja_con1 = ja_con[n_s:,:n_s]
        _,indices = torch.sort(ja_con1, descending=True)
        del net

        return indices






    def finetunning(self, img_spt, mask_spt, img_qry, mask_qry):
        assert len(img_spt.shape) == 4
        net = deepcopy(self.net)
        logits,_ = net(img_spt, net.parameters(), bn_training=True)
        loss = criterion(logits, mask_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        for k in range(1, self.update_step_test):
            mask_hat,_ = net(img_spt, fast_weights, bn_training=True)
            loss = criterion(mask_hat, mask_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            if k == self.update_step_test - 1:
                with torch.no_grad():
                    mask_hat,_ = net(img_qry, fast_weights, bn_training=True)
                    pred_qry = torch.sigmoid(mask_hat)
                    pred_qry = torch.where(pred_qry > 0.5, 1, 0)
                    iou_test = iou(mask_qry, pred_qry).item()



        del net
        return pred_qry,iou_test





def main():
    pass


if __name__ == '__main__':
    main()









