import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def dice_loss (logit, target):
    n, c, h, w = logit.size()
    loss = 0

    for i in range (n):
        #l = np.ones((h, w))
        #l = torch.tensor(l).cuda()
        probs = torch.sigmoid(logit[i])
        label1 = target[i].view(1, h*w)
        #label0 = l - target[i]
        output1 = probs.view(1, h*w)
        #output0 = l - probs
        #intersection0 = output0 * label0
        intersection1 = output1 * label1
        #DSC0 = (2 * torch.abs(torch.sum(intersection0)) + 0.0001) / (
                    #torch.abs(torch.sum(output0)) + torch.sum(label0) + 0.0001)
        DSC1 = (2 * torch.abs(torch.sum(intersection1)) + 1) / (
                    torch.abs(torch.sum(output1)) + torch.sum(label1) + 1)
        loss_i = 1 - DSC1
        loss = loss + loss_i
    #loss = loss

    return loss

def criterion(logit, target):
    n, c, h, w = logit.size()
    loss = 0

    for i in range(n):
        # l = np.ones((h, w))
        # l = torch.tensor(l).cuda()
        probs = torch.sigmoid(logit[i])
        label1 = target[i].view(1, h * w)
        # label0 = l - target[i]
        output1 = probs.view(1, h * w)
        # output0 = l - probs
        # intersection0 = output0 * label0
        intersection1 = output1 * label1
        # DSC0 = (2 * torch.abs(torch.sum(intersection0)) + 0.0001) / (
        # torch.abs(torch.sum(output0)) + torch.sum(label0) + 0.0001)
        DSC1 = (2 * torch.sum(intersection1) +0.01) / (torch.sum(output1) + torch.sum(label1) +0.01)
        loss_i = 1 - DSC1
        loss = loss + loss_i
    loss = loss/n

    return loss


def metric (logit, target):
    n, c, h, w = logit.size()
    loss = 0
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    for i in range(n):


        #l = torch.ones(h,w).to(device)
        probs = torch.sigmoid(logit[i])
        label1 = target[i].view(1, h * w)
        #label0 = (l - target[i]).view(1, h * w)
        output1 = probs.view(1, h * w)
        loss_i = torch.sum(2 * torch.abs(output1 * label1 / (output1 + label1)))
        loss = loss + loss_i

    return loss

def focal (logit, target):
    n, c, h, w = logit.size()
    gamma = 2
    pred = torch.sigmoid(logit)
    preds = torch.clamp(pred, 0.001, 0.999)
    loss = 0
    for i in range(n):
        #l = torch.ones(h, w).cuda()
        log1 = torch.log(preds[i])
        log0 = torch.log(1-preds[i])
        alpha = torch.sum(target[i]) / (h * w)
        loss_i = - (1-alpha) * ((1 - preds[i]) **gamma) * target[i] * log1 - alpha * (preds[i] **gamma) * (1 - target[i]) * log0
        loss_i = loss_i.sum()
        loss = loss +loss_i

    return loss



if __name__ == "__main__":

    a = torch.rand(2, 1, 512, 512).cuda()
    b = torch.rand(2, 1, 512, 512).cuda()
    loss = metric(a, b)
    print(loss)







