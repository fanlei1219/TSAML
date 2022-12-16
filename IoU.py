import torch
from torch.autograd import Function
import numpy as np
import math

def generate_matrix(gt_image, pre_image):
    num_class = 2
    mask = (gt_image >=0)&(gt_image < num_class)
    label = num_class * gt_image[mask].astype('int')+pre_image[mask].astype('int')
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix



def iou(inputs,targets):
    if inputs.is_cuda:
        miou = torch.FloatTensor(1).cuda().zero_()
    else:
        miou = torch.FloatTensor(1).zero_()
    n,c,h,w = inputs.size()

    for i in range(n):
        input = inputs[i]#.numpy()
        target = targets[i]#.numpy()
        input= np.where(input > 0.5, 1, 0)
        target= np.where(target > 0.5, 1, 0)
        confusion_matrix = generate_matrix(input, target)
        MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
        MIoU = np.mean(MIoU)
        miou = MIoU + miou

    miou = miou/n
    return(miou)

















class IoU(Function):
    def forward(self, inputs, target):
        self.save_for_backward(inputs, target)
        #eps = 0.0001

        inputs = np.where(inputs > 0, 1, 0)
        target = np.where(target > 0, 1, 0)


        inter = np.logical_and(inputs, target).sum()
        union = inputs.sum() + target.sum() - inter

        iou_each = inter / union
        return iou_each




def mmmiou(inputs, target):
    """Iou for batches"""
    if inputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(inputs, target)):
        each_iou = IoU().forward(c[0], c[1])
        if math.isnan(each_iou):
            each_iou = 0

        s = s + each_iou

    return s / (i + 1)
