import numpy as np
import torch
import torch.nn as nn
# from utils.loss import cross_entropy_2d
# import cv2
import torch.nn.functional as F
import torch.sparse as sparse


def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w [4,256,256],torch.Size([4, 256, 256])
    dice     = 0
    dice_arr = [] #每个类别的预测Dice结果,也包含背景的
    each_class_number = [] #统计gt下每个类别的像素总数
    eps      = 1e-7

    #遍历每个类别
    for i in range(1,n_class):

        A = (pred  == i) #预测与每个类别之间的布尔矩阵
        B = (label == i) #金标准与每个类别之间的布尔矩阵

        each_class_number.append(torch.sum(B).cpu().data.numpy())

        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

# def loss_calc(pred,label,cfg,device):
#
#     '''
#     This function returns cross entropy loss for semantic segmentation
#     '''
#     # pred shape is batch * c * h * w
#     # label shape is b*h*w
#     label = label.long().cuda()
#     return cross_entropy_2d(pred, label,cfg,device)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.max_iterations, cfg.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    #cfg.TRAIN.LEARNING_RATE_D = 1e-4
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.base_lr)

def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.gan_base_lr)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)

def sel_prob_2_entropy(prob):
    n, c, h, w = prob.size()
    weighted_self_info = -torch.mul(prob, torch.log2(prob + 1e-30)) / c
    entropy            = torch.sum(weighted_self_info,dim=1) #N*C*H*W
    # mean_entropy       = torch.sum(entropy,dim=[1,2])
    return entropy



def mpcl_loss_calc(feas,labels,class_center_feas,loss_func,
                               pixel_sel_loc=None,tag='source'):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    # print('feas.shape,',feas.shape) #torch.Size([4, 2048, 17, 17])
    # print('labels.shape,',labels.shape) #torch.Size([4, 128,128])
    # print('class_center_feas.shape,',class_center_feas.shape) #torch.Size([2,2048])


    n,c,fea_h,fea_w = feas.size() ##torch.Size([4, 2048, 17, 17])
    if tag == 'source':
        labels      = labels.float()
        labels      = F.interpolate(labels, size=fea_w, mode='nearest') #torch.Size([4, 128, 17])
        labels      = labels.permute(0,2,1).contiguous() ##torch.Size([4, 17, 128]) b,w,h
        labels      = F.interpolate(labels, size=fea_h, mode='nearest')  ##torch.Size([4, 17, 17])
        labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w ##torch.Size([4, 17, 17])

    labels  = labels.cuda()
    labels  = labels.view(-1).long() #4x17x17, [1156]

    feas = torch.nn.functional.normalize(feas,p=2,dim=1) #torch.Size([4, 2048, 17, 17])
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c torch.Size([4, 17, 17, 2048])
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c torch.Size([1156, 2048])
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c torch.Size([1156, 1, 2048])

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1) #torch.Size([2,2048])
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class #torch.Size([2048,2])


    #1.torch.Size([1156, 1, 2048]);2.[1156];3.#torch.Size([2048,2]);
    loss =  loss_func(feas,labels,class_center_feas,
                                                    pixel_sel_loc=pixel_sel_loc)

    if torch.isnan(loss).any():
        return torch.tensor(0., device=loss.device, dtype=loss.dtype)
    else:
        return loss


    # return loss


