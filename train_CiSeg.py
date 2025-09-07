import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.func import *
import torchvision
from utils.visual import decode_seg_map_sequence_tr

from dataloaders import utils
from dataloaders.dataset import MRBaseDataSets, CTBaseDataSets, RandomGenerator,BaseDataSets
from networks.net_factory import net_factory
from networks.discriminator import get_discriminatord
from utils.losses import DiceLoss,DiceLossSDM,GCELoss,SCELoss,FocalLoss,GeneralizedDiceLoss,PixelCLLossSrc,PixelCLLossTrg
from val_2D import test_single_volume,test_single_volumeall
import os

cpu_num = 2
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                  default='./data/cardiac', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CiSeg', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='deeplabv2_ours', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--d_type',  default='PatchGAN',
                    help='batch_size per gpu')
parser.add_argument('--power', type=int, default=0.9,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=2.5e-4,
                    help='segmentation network learning rate')
parser.add_argument('--gan_base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
args = parser.parse_args()


def DiceLoss_DSV(out_group, target, aux_weights=[0.4, 0.6, 0.8]):
  criterion = DiceLossSDM(4)
  loss = 0
  for i, out in enumerate(out_group):
    loss += aux_weights[i] * criterion(out, target.unsqueeze(1))
  return loss


def CrossEntropyLoss_DSV(out_group, target, aux_weights=[0.4, 0.6, 0.8]):
  # criterion = nn.CrossEntropyLoss(torch.tensor([1., 2.]).cuda())
  criterion = CrossEntropyLoss()
  loss = 0
  for i, out in enumerate(out_group):
    loss += aux_weights[i] * criterion(out, target.long())
    # print(loss)
  return loss


def label_downsample(labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()

    return labels


def update_class_center_iter(cla_src_feas,batch_src_labels,class_center_feas,m):



    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels.cuda()
    n,c,fea_h,fea_w    = batch_src_feas.size()

    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w

    batch_class_center_fea_list = []

    for i in range(5):

        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w,
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c,
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])

        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num

        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)


    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c，torch.Size([5, 2048])
    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas

    return class_center_feas



def update_class_center_iter_tar(cla_feas_trg_l, pred_trg_main,class_center_feas_main_tar,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''

    pred_trg_main_soft = F.softmax(pred_trg_main, dim=1)
    batch_trg_feas     = cla_feas_trg_l.detach()
    pred_trg_main_soft   = pred_trg_main_soft.cuda()

    n,c,fea_h,fea_w    = batch_trg_feas.size()
    _,num_class,_,_ = pred_trg_main_soft.size()
    c_fea1 = batch_trg_feas.shape[1]

    batch_trg_feas = batch_trg_feas.permute(0, 2, 3, 1).reshape(-1, c_fea1)

    k = fea_h * fea_w //15 #72
    k = int(k)
    top_values, top_indices = torch.topk(pred_trg_main_soft.transpose(0, 1).reshape(num_class, -1),
                                         k=k, dim=-1)

    batch_class_center_fea_list = []

    for i in range(5):

        top_fea = batch_trg_feas[top_indices[i]]
        batch_class_center_fea = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)

    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c，
    class_center_feas = m * class_center_feas_main_tar + (1-m) * batch_class_center_feas

    return class_center_feas


def train(args, snapshot_path):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_l = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model_b = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    norm_layer = nn.BatchNorm2d
    d_aux = get_discriminatord(args.d_type, 5, norm_layer, init_type='normal').cuda()
    d_main = get_discriminatord(args.d_type, 5, norm_layer, init_type='normal').cuda()
    # d_aux_res = get_discriminatord(args.d_type, 3, norm_layer, init_type='normal').cuda()
    d_main_res = get_discriminatord(args.d_type, 3, norm_layer, init_type='normal').cuda()

    db_train_mr = MRBaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))
    db_train_ct = CTBaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))
    db_val_ct = CTBaseDataSets(base_dir=args.root_path, split="val")
    # db_val_mr = MRBaseDataSets(base_dir=args.root_path, split="val")
    #
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader_mr = DataLoader(db_train_mr, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_ct = DataLoader(db_train_ct, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader_ct = DataLoader(db_val_ct, batch_size=1, shuffle=False,
                           num_workers=0)
    # valloader_mr = DataLoader(db_val_mr, batch_size=1, shuffle=False,
    #                           num_workers=0)

    model_l.train()
    model_b.train()
    d_aux.train()
    d_main.train()
    # d_aux_res.train()
    d_main_res.train()

    optimizer_l = optim.SGD(model_l.optim_parameters(base_lr), lr=base_lr,
                          momentum=0.9, weight_decay=0.0005)

    optimizer_b = optim.SGD(model_b.optim_parameters(base_lr), lr=base_lr,
                          momentum=0.9, weight_decay=0.0005)


    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=args.gan_base_lr,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=args.gan_base_lr,
                                  betas=(0.9, 0.99))
    optimizer_d_main_res = optim.Adam(d_main_res.parameters(), lr=args.gan_base_lr,
                                      betas=(0.9, 0.99))
    # optimizer_d_aux_res = optim.Adam(d_aux_res.parameters(), lr=args.gan_base_lr,
    #                                   betas=(0.9, 0.99))

    # ce_loss = CrossEntropyLoss()
    ce_loss = CrossEntropyLoss(reduction='none')
    # bias_criterion = GCELoss(num_classes=5)
    # bias_criterion = SCELoss(num_classes=5)
    bias_criterion = FocalLoss()
    bias_dice_criterion = GeneralizedDiceLoss()
    dice_loss = DiceLoss(num_classes)
    cl_criterion = PixelCLLossSrc(args)
    cl_criterion_trg = PixelCLLossTrg(args)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} train mr iterations per epoch".format(len(trainloader_mr)))
    logging.info("{} train ct iterations per epoch".format(len(trainloader_ct)))
    logging.info("{} val ct iterations per epoch".format(len(valloader_ct)))
    # logging.info("{} val mr iterations per epoch".format(len(valloader_mr)))

    interp = nn.Upsample(size=(256, 256), mode='bilinear',align_corners=True)

    source_label = 0
    targte_label = 1

    # iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    # iterator = tqdm(range(max_epoch), ncols=70)

    strain_loader_iter = enumerate(trainloader_mr)
    trgtrain_loader_iter = enumerate(trainloader_ct)

    # strain_loader_iter = enumerate(trainloader_ct)
    # trgtrain_loader_iter = enumerate(trainloader_mr)

    class_center_feas_main = torch.zeros(5, 2048, dtype=torch.float).cuda()
    # class_center_feas_aux = torch.zeros(5, 1024, dtype=torch.float).cuda()

    class_center_feas_main_tar = torch.zeros(5, 2048, dtype=torch.float).cuda()
    # class_center_feas_aux_tar = torch.zeros(5, 1024, dtype=torch.float).cuda()

    for iter_num in tqdm(range(max_iterations+1)):

        model_l.train()
        model_b.train()
        d_main.train()
        d_aux.train()
        d_main_res.train()
        # d_aux_res.train()

        # reset optimizers
        optimizer_l.zero_grad()
        optimizer_b.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_main_res.zero_grad()
        # optimizer_d_aux_res.zero_grad()

        adjust_learning_rate(optimizer_l, iter_num, args)
        adjust_learning_rate(optimizer_b, iter_num, args)
        adjust_learning_rate_discriminator(optimizer_d_aux, iter_num, args)
        adjust_learning_rate_discriminator(optimizer_d_main, iter_num, args)
        adjust_learning_rate_discriminator(optimizer_d_main_res, iter_num, args)
        # adjust_learning_rate_discriminator(optimizer_d_aux_res, iter_num, args)

        ###########################################################################

        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        for param in d_main_res.parameters():
            param.requires_grad = False
        # for param in d_aux_res.parameters():
        #     param.requires_grad = False

        try:
            _, sampled_batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(trainloader_mr)
            # strain_loader_iter = enumerate(trainloader_ct)
            _, sampled_batch = strain_loader_iter.__next__()

        volume_batch_src, label_batch_src = sampled_batch['image'], sampled_batch['label']  #torch.Size([24, 3, 128, 128]),torch.Size([24, 128, 128])
        volume_batch_src, label_batch_src = volume_batch_src.cuda(), label_batch_src.cuda()

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trainloader_ct)
            # trgtrain_loader_iter = enumerate(trainloader_mr)
            _, batch = trgtrain_loader_iter.__next__()

        volume_batch_trg, label_batch_trg = batch['image'], batch['label']
        volume_batch_trg, label_batch_trg = volume_batch_trg.cuda(), label_batch_trg.cuda()

        if volume_batch_trg.size()[0] < volume_batch_src.size()[0]:

            volume_batch_src = volume_batch_src[:volume_batch_trg.size()[0], :, :, :]
            label_batch_src = label_batch_src[:volume_batch_trg.size()[0], :, :]

        if volume_batch_src.size()[0] < volume_batch_trg.size()[0]:

            volume_batch_trg = volume_batch_trg[:volume_batch_src.size()[0], :, :, :]
            # label_batch_trg = label_batch_trg[:volume_batch_src.size()[0], :, :]

        #####################################源域的###############################################

        cla_feas_src_aux_l, cla_feas_src_l = model_l(volume_batch_src)  # Zi
        cla_feas_src_aux_b, cla_feas_src_b = model_b(volume_batch_src) #Zb

        ###############################################

        cla_feas_src_conflict = torch.cat((cla_feas_src_l, cla_feas_src_b.detach()), dim=1)
        cla_feas_src_align = torch.cat((cla_feas_src_l.detach(), cla_feas_src_b), dim=1)

        cla_feas_src_aux_conflict = torch.cat((cla_feas_src_aux_l, cla_feas_src_aux_b.detach()), dim=1)
        cla_feas_src_aux_align = torch.cat((cla_feas_src_aux_l.detach(), cla_feas_src_aux_b), dim=1)

        ###############################################
        class_center_feas_main = update_class_center_iter(cla_feas_src_l, label_batch_src, class_center_feas_main, m=0.20)
        # class_center_feas_aux = update_class_center_iter(cla_feas_src_aux_l, label_batch_src, class_center_feas_aux, m=0.20)
        #########################################################


        pred_src_main_conflict = model_l.layer6(cla_feas_src_conflict)
        pred_src_main_align = model_b.layer6(cla_feas_src_align)

        pred_src_aux_conflict = model_l.layer7(cla_feas_src_aux_conflict)
        pred_src_aux_align = model_b.layer7(cla_feas_src_aux_align)

        ##################################################

        pred_src_main_conflict = interp(pred_src_main_conflict)
        pred_src_main_align = interp(pred_src_main_align)
        pred_src_main = pred_src_main_conflict #causal
        pred_src_main_b = pred_src_main_align #bias

        # pred_src_main_res_soft_l = pred_src_main.clone().detach()
        # pred_src_main_res_soft_l = torch.argmax(pred_src_main_res_soft_l, dim=1)

        # pred_src_main_res_soft_b = pred_src_main_b.clone().detach()
        # pred_src_main_res_soft_b = torch.argmax(pred_src_main_res_soft_b, dim=1)

        # pred_src_main_res = torch.abs(pred_src_main - pred_src_main_b)
        # pred_src_main_res = torch.abs(pred_src_main_res_soft_l - pred_src_main_res_soft_b) #torch.Size([4, 128, 128])
        # pred_src_main_res = pred_src_main_res.unsqueeze(1).repeat(1, 3, 1, 1)

        pred_src_aux_conflict = interp(pred_src_aux_conflict)  # torch.Size([24, 5, 128, 128])
        pred_src_aux_align = interp(pred_src_aux_align)
        pred_src_aux = pred_src_aux_conflict ##causal
        pred_src_aux_b = pred_src_aux_align #bias

        ###################################################

        pred_src_main_conflict_soft = torch.softmax(pred_src_main_conflict, dim=1) #
        pred_src_main_align_soft = torch.softmax(pred_src_main_align, dim=1)

        pred_src_aux_conflict_soft = torch.softmax(pred_src_aux_conflict, dim=1)
        pred_src_aux_align_soft = torch.softmax(pred_src_aux_align, dim=1)

        loss_dice_dis_conflict = dice_loss(pred_src_main_conflict_soft, label_batch_src.unsqueeze(1)).detach()
        loss_dice_dis_align = dice_loss(pred_src_main_align_soft, label_batch_src.unsqueeze(1)).detach()

        loss_dice_aux_dis_conflict = dice_loss(pred_src_aux_conflict_soft, label_batch_src.unsqueeze(1)).detach()
        loss_dice_aux_dis_align = dice_loss(pred_src_aux_align_soft, label_batch_src.unsqueeze(1)).detach()

        loss_weight_dice = loss_dice_dis_align / (loss_dice_dis_align + loss_dice_dis_conflict + 1e-8)
        loss_dis_conflict_dice = dice_loss(pred_src_main_conflict_soft, label_batch_src.unsqueeze(1)) * loss_weight_dice
        loss_dis_align_dice = bias_dice_criterion(pred_src_main_align_soft, label_batch_src.unsqueeze(1))

        loss_aux_weight_dice = loss_dice_aux_dis_align / (loss_dice_aux_dis_align + loss_dice_aux_dis_conflict + 1e-8)
        loss_aux_dis_conflict_dice = dice_loss(pred_src_aux_conflict_soft, label_batch_src.unsqueeze(1)) * loss_aux_weight_dice
        loss_aux_dis_align_dice = bias_dice_criterion(pred_src_aux_align_soft, label_batch_src.unsqueeze(1))

        ####################################################


        loss_dis_conflict  = ce_loss(pred_src_main_conflict, label_batch_src[:].long()).detach()
        loss_dis_align  = ce_loss(pred_src_main_align, label_batch_src[:].long()).detach()

        loss_dis_aux_conflict = ce_loss(pred_src_aux_conflict, label_batch_src[:].long()).detach()
        loss_dis_aux_align = ce_loss(pred_src_aux_align, label_batch_src[:].long()).detach()

        loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
        loss_dis_conflict = ce_loss(pred_src_main_conflict, label_batch_src[:].long()) * loss_weight
        loss_dis_align = bias_criterion(pred_src_main_align, label_batch_src)

        loss_aux_weight = loss_dis_aux_align / (loss_dis_aux_align + loss_dis_aux_conflict + 1e-8)
        loss_dis_aux_conflict = ce_loss(pred_src_aux_conflict, label_batch_src[:].long()) * loss_aux_weight
        loss_dis_aux_align = bias_criterion(pred_src_aux_align, label_batch_src)

        ###############################################################################

        indices = np.random.permutation(cla_feas_src_b.size(0))
        z_b_swap = cla_feas_src_b[indices]
        label_swap =  label_batch_src[indices]

        indices_aux = np.random.permutation(cla_feas_src_aux_b.size(0))
        z_b_swap_aux = cla_feas_src_aux_b[indices_aux]
        label_swap_aux = label_batch_src[indices_aux]

        z_mix_conflict = torch.cat((cla_feas_src_l, z_b_swap.detach()), dim=1)
        z_mix_align = torch.cat((cla_feas_src_l.detach(), z_b_swap), dim=1)

        z_mix_conflict_aux = torch.cat((cla_feas_src_aux_l, z_b_swap_aux.detach()), dim=1)
        z_mix_align_aux = torch.cat((cla_feas_src_aux_l.detach(), z_b_swap_aux), dim=1)

        pred_mix_conflict = model_l.layer6(z_mix_conflict)
        pred_mix_align = model_b.layer6(z_mix_align)

        pred_mix_conflict_aux = model_l.layer7(z_mix_conflict_aux)
        pred_mix_align_aux = model_b.layer7(z_mix_align_aux)

        pred_mix_conflict = interp(pred_mix_conflict)
        pred_mix_align = interp(pred_mix_align)

        pred_mix_conflict_aux = interp(pred_mix_conflict_aux)
        pred_mix_align_aux = interp(pred_mix_align_aux)

        loss_swap_conflict = ce_loss(pred_mix_conflict, label_batch_src[:].long()) * loss_weight
        loss_swap_align = bias_criterion(pred_mix_align, label_swap)

        loss_swap_conflict_aux = ce_loss(pred_mix_conflict_aux,label_batch_src[:].long()) * loss_aux_weight
        loss_swap_align_aux = bias_criterion(pred_mix_align_aux, label_swap_aux)

        pred_mix_conflict_soft = torch.softmax(pred_mix_conflict, dim=1)
        pred_mix_align_soft = torch.softmax(pred_mix_align, dim=1)

        pred_mix_conflict_aux_soft = torch.softmax(pred_mix_conflict_aux, dim=1)
        pred_mix_align_aux_soft = torch.softmax(pred_mix_align_aux , dim=1)

        loss_swap_conflict_dice = dice_loss(pred_mix_conflict_soft, label_batch_src.unsqueeze(1)) * loss_weight_dice
        loss_swap_align_dice = bias_dice_criterion(pred_mix_align_soft, label_swap.unsqueeze(1))

        loss_swap_conflict_aux_dice = dice_loss(pred_mix_conflict_aux_soft, label_batch_src.unsqueeze(1)) * loss_aux_weight_dice
        loss_swap_align_aux_dice = bias_dice_criterion(pred_mix_align_aux_soft, label_swap_aux.unsqueeze(1))

        #######################################################################################

        loss_dis = loss_dis_conflict.mean() + loss_dis_align  + loss_dis_conflict_dice + loss_dis_align_dice
        loss_swap = loss_swap_conflict.mean() + loss_swap_align + loss_swap_conflict_dice + loss_swap_align_dice

        loss_dis_aux = loss_dis_aux_conflict.mean() + loss_dis_aux_align + loss_aux_dis_conflict_dice + loss_aux_dis_align_dice
        loss_swap_aux = loss_swap_conflict_aux.mean() + loss_swap_align_aux + loss_swap_conflict_aux_dice + loss_swap_align_aux_dice

        ##############################################################################

        cla_feas_trg_aux_l, cla_feas_trg_l = model_l(volume_batch_trg)
        cla_feas_trg_aux_b, cla_feas_trg_b = model_b(volume_batch_trg)

        cla_feas_trg_l_main_all = torch.cat((cla_feas_trg_l, cla_feas_trg_b), dim=1)
        cla_feas_trg_l_aux_all = torch.cat((cla_feas_trg_aux_l, cla_feas_trg_aux_b), dim=1)

        pred_trg_main = model_l.layer6(cla_feas_trg_l_main_all)
        pred_trg_aux = model_l.layer7(cla_feas_trg_l_aux_all)

        pred_trg_main_b = model_b.layer6(cla_feas_trg_l_main_all)
        pred_trg_main_b = interp(pred_trg_main_b)

        pred_trg_aux_b = model_b.layer7(cla_feas_trg_l_aux_all)
        pred_trg_aux_b = interp(pred_trg_aux_b)

        class_center_feas_main_tar = update_class_center_iter_tar(cla_feas_trg_l, pred_trg_main,
                                                             class_center_feas_main_tar.detach(), m=0.20)

        # class_center_feas_aux_tar = update_class_center_iter_tar(cla_feas_trg_aux_l, pred_trg_aux,
        #                                                      class_center_feas_aux_tar.detach(), m=0.20)


        ###############################################################

        pred_trg_aux = interp(pred_trg_aux)
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        loss_adv_trg_aux = bce_loss(d_out_aux, source_label)

        pred_trg_main = interp(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)

        loss_adv = 0.003 * loss_adv_trg_main + 0.00002 * loss_adv_trg_aux

        ############################################################################
        pred_src_main_res_soft_l = pred_src_main.clone().detach()
        pred_src_main_res_soft_l = torch.argmax(pred_src_main_res_soft_l, dim=1)

        pred_src_main_res_soft_b = pred_src_main_b.clone().detach()
        pred_src_main_res_soft_b = torch.argmax(pred_src_main_res_soft_b, dim=1)

        pred_trg_main_res_soft_l = pred_trg_main.clone().detach()
        pred_trg_main_res_soft_l = torch.argmax(pred_trg_main_res_soft_l, dim=1)
        #
        pred_trg_main_res_soft_b = pred_trg_main_b.clone().detach()
        pred_trg_main_res_soft_b = torch.argmax(pred_trg_main_res_soft_b, dim=1)

        pred_src_main_res_l = torch.abs(pred_src_main_res_soft_l - pred_trg_main_res_soft_l)
        pred_src_main_res_l = pred_src_main_res_l.unsqueeze(1).repeat(1, 3, 1, 1)
        #
        pred_trg_main_res = torch.abs(pred_src_main_res_soft_b - pred_trg_main_res_soft_b)
        pred_trg_main_res = pred_trg_main_res.unsqueeze(1).repeat(1, 3, 1, 1)

        loss_adv_trg_main_res = bce_loss(pred_trg_main_res.float(), source_label)

        loss_res_main = loss_adv_trg_main_res

        ############################

        # pred_src_aux_res_soft_l = pred_src_aux.clone().detach()
        # pred_src_aux_res_soft_l = torch.argmax(pred_src_aux_res_soft_l, dim=1)  # torch.Size([4, 128, 128])
        #
        # pred_src_aux_res_soft_b = pred_src_aux_b.clone().detach()
        # pred_src_aux_res_soft_b = torch.argmax(pred_src_aux_res_soft_b, dim=1)  # torch.Size([4, 128, 128])
        #
        # pred_trg_aux_res_soft_l = pred_trg_aux.clone().detach()
        # pred_trg_aux_res_soft_l = torch.argmax(pred_trg_aux_res_soft_l, dim=1)
        # #
        # pred_trg_aux_res_soft_b = pred_trg_aux_b.clone().detach()
        # pred_trg_aux_res_soft_b = torch.argmax(pred_trg_aux_res_soft_b, dim=1)
        #
        # pred_src_aux_res = torch.abs(pred_src_aux_res_soft_l - pred_trg_aux_res_soft_l)
        # pred_src_aux_res = pred_src_aux_res.unsqueeze(1).repeat(1, 3, 1, 1)
        # #
        # pred_trg_aux_res = torch.abs(pred_src_aux_res_soft_b - pred_trg_aux_res_soft_b)
        # pred_trg_aux_res = pred_trg_aux_res.unsqueeze(1).repeat(1, 3, 1, 1)
        #
        # loss_adv_trg_aux_res = bce_loss(pred_trg_aux_res.float(), source_label)
        #
        # loss_res_aux = loss_adv_trg_aux_res
        #
        # loss_res = loss_res_main + loss_res_aux

        loss_res = loss_res_main

        ##########################################################################
        loss_cl_src_main,_ = cl_criterion(semantic_prototype=class_center_feas_main,
                                        src_feat=cla_feas_src_l,
                                        src_mask=label_batch_src)
        loss_cl_mian = loss_cl_src_main

        # loss_cl_src_aux, _ = cl_criterion(semantic_prototype=class_center_feas_aux,
        #                                    src_feat=cla_feas_src_aux_l,
        #                                    src_mask=label_batch_src)
        # loss_cl_aux = loss_cl_src_aux

        # loss_cl = loss_cl_mian + loss_cl_aux
        loss_cl = loss_cl_mian

        ########################################################################

        loss_cl_tar_main, _ = cl_criterion_trg(semantic_prototype=class_center_feas_main_tar,
                                                        src_feat=cla_feas_src_l,
                                                        src_mask=label_batch_src)
        #
        # loss_cl_tar_aux, _ = cl_criterion_trg(semantic_prototype=class_center_feas_aux_tar,
        #                                        src_feat=cla_feas_src_aux_l,
        #                                        src_mask=label_batch_src)
        #
        # loss_cl_tar = loss_cl_tar_main + loss_cl_tar_aux
        loss_cl_tar = loss_cl_tar_main

        ############################################################################

        loss_seg = loss_dis + 5.0 * loss_swap + loss_dis_aux + 5.0 * loss_swap_aux
        # loss_seg = loss_dis + 5.0 * loss_swap

        loss_cl_all = 0.40 * loss_cl + 0.40 * loss_cl_tar

        loss_res_all = 0.85 * loss_res

        loss_adv_all = loss_adv


        loss = loss_seg + loss_adv_all + loss_cl_all + loss_res_all

        loss.backward()
        #############################################################################

        # Train  scriminator networks

        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        for param in d_main_res.parameters():
            param.requires_grad = True
        # for param in d_aux_res.parameters():
        #     param.requires_grad = True

        pred_src_aux = pred_src_aux.detach()  # 源域的输出结果
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux, dim=1)))
        loss_d_aux = bce_loss(d_out_aux, source_label)  # 源域就是源域
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()

        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        pred_src_main_res_l = pred_src_main_res_l.detach()
        d_out_main_res = d_main_res(F.softmax(pred_src_main_res_l.float(), dim=1))
        loss_d_main_res = bce_loss(d_out_main_res, source_label)
        loss_d_main_res = loss_d_main_res / 2
        loss_d_main_res.backward()

        # pred_src_aux_res = pred_src_aux_res.detach()
        # d_out_aux_res = d_aux_res(F.softmax(pred_src_aux_res.float(), dim=1))
        # loss_d_aux_res = bce_loss(d_out_aux_res, source_label)
        # loss_d_aux_res = loss_d_aux_res / 2
        # loss_d_aux_res.backward()

        # second we train with target
        pred_trg_aux = pred_trg_aux.detach()
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        loss_d_aux = bce_loss(d_out_aux, targte_label)
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()

        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, targte_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        pred_trg_main_res = pred_trg_main_res.detach()
        d_out_main_trg_res = d_main_res(F.softmax(pred_trg_main_res.float(), dim=1))
        loss_out_main_trg_res = bce_loss(d_out_main_trg_res, targte_label)
        loss_out_main_trg_res = loss_out_main_trg_res / 2
        loss_out_main_trg_res.backward()

        # pred_trg_aux_res = pred_trg_aux_res.detach()
        # d_out_aux_trg_res = d_aux_res(F.softmax(pred_trg_aux_res.float(), dim=1))
        # loss_out_aux_trg_res = bce_loss(d_out_aux_trg_res, targte_label)
        # loss_out_aux_trg_res = loss_out_aux_trg_res / 2
        # loss_out_aux_trg_res.backward()

        optimizer_l.step()
        optimizer_b.step()
        optimizer_d_aux.step()
        optimizer_d_main.step()
        optimizer_d_main_res.step()
        # optimizer_d_aux_res.step()

        # writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/loss_seg', loss_seg, iter_num)
        writer.add_scalar('info/loss_adv_all', loss_adv_all, iter_num)
        writer.add_scalar('info/loss_cl_all', loss_cl_all, iter_num)
        writer.add_scalar('info/loss_res_all', loss_res_all, iter_num)

        logging.info('iteration %d : loss : %f, loss_seg: %f, loss_adv_all: %f, loss_cl_all: %f, loss_res_all: %f' %
                     (iter_num, loss.item(), loss_seg.item(), loss_adv_all.item(),loss_cl_all.item(), loss_res_all.item()))

        if iter_num % 200 == 0:

            writer.add_scalar('info_200/total_loss', loss, iter_num)
            writer.add_scalar('info_200/loss_seg', loss_seg, iter_num)
            writer.add_scalar('info_200/loss_adv_all', loss_adv_all, iter_num)
            writer.add_scalar('info_200/loss_cl_all', loss_cl_all, iter_num)
            writer.add_scalar('info_200/loss_res_all', loss_res_all, iter_num)


        if iter_num > 0 and iter_num % 200 == 0:

            model_l.eval()
            model_b.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader_ct):

                metric_i = test_single_volumeall(
                        sampled_batch["image"], sampled_batch["label"], model_l, model_b, classes=num_classes)

                metric_list += np.array(metric_i)

            metric_list = metric_list / len(db_val_ct)
            # metric_list = metric_list / len(db_val_mr)
            print('metric_list,', metric_list)

            print('metric_list[MYO],', metric_list[0, 0])
            print('metric_list[LAC],', metric_list[1, 0])
            print('metric_list[LVC],', metric_list[2, 0])
            print('metric_list[AA],', metric_list[3, 0])
            #
            # print('metric_list[Liver],', metric_list[0, 0])
            # print('metric_list[Lk],', metric_list[1, 0])
            # print('metric_list[Rk],', metric_list[2, 0])
            # print('metric_list[Spl],', metric_list[3, 0])


            for class_i in range(num_classes-1):
                writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
            print(np.mean(metric_list, axis=0))

            performance = np.mean(metric_list, axis=0)[0]

            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:

                class_center_feas_save_dir = snapshot_path + '/feas_main'
                os.makedirs(class_center_feas_save_dir, exist_ok=True)
                class_center_feas_save_pth = f'{class_center_feas_save_dir}/class_center_feas_model_{iter_num}.npy'

                class_center_feas_npy = class_center_feas_main.cpu().detach().numpy()
                np.save(class_center_feas_save_pth, class_center_feas_npy)

                # class_center_feas_save_dir_aux = snapshot_path + '/feas_aux'
                # os.makedirs(class_center_feas_save_dir_aux, exist_ok=True)
                # class_center_feas_save_pth_aux = f'{class_center_feas_save_dir_aux}/class_center_feas_model_{iter_num}.npy'

                best_performance = performance

                save_mode_path_l = os.path.join(snapshot_path,
                                                  'model_l_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                save_best_l = os.path.join(snapshot_path,
                                             'model_l_{}_best_model.pth'.format(args.model))
                torch.save(model_l.state_dict(), save_mode_path_l)
                torch.save(model_l.state_dict(), save_best_l)

                save_mode_path_b = os.path.join(snapshot_path,
                                                'model_b_iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                save_best_b = os.path.join(snapshot_path,
                                           'model_b_{}_best_model.pth'.format(args.model))
                torch.save(model_b.state_dict(), save_mode_path_b)
                torch.save(model_b.state_dict(), save_best_b)


                save_D_aux_path = os.path.join(snapshot_path,
                                              'iter_{}_D_aux.pth'.format(iter_num))
                torch.save(d_aux.state_dict(), save_D_aux_path)


                save_D_main_path = os.path.join(snapshot_path,
                                               'iter_{}_D_main.pth'.format(iter_num))
                torch.save(d_main.state_dict(), save_D_main_path)

            logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
            model_l.train()
            model_b.train()

        if iter_num % 2000 == 0:

                save_mode_path_l = os.path.join(snapshot_path, 'model_l_iter_' + str(iter_num) + '.pth')
                save_mode_path_b = os.path.join(snapshot_path, 'model_b_iter_' + str(iter_num) + '.pth')
                torch.save(model_l.state_dict(), save_mode_path_l)
                torch.save(model_b.state_dict(), save_mode_path_b)
                logging.info("save model to {}".format(save_mode_path_l))
                logging.info("save model to {}".format(save_mode_path_b))

        if iter_num >= max_iterations:
                break

    writer.close()
    return "Training Finished!"



if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./experiments/heart/mr2ct/{}/{}".format(args.exp, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
