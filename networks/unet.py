# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np
import torch.nn.functional as F



def masked_average_pooling(feature, mask):
    #print(feature.shape[-2:])
    # print(feature.shape)  #torch.Size([24, 32, 256, 256])
    # print(mask.shape) #torch.Size([24, 1, 256, 256])

    mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True) # #torch.Size([24, 1, 256, 256])

    #print((feature*mask).shape)
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                     / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature


def batch_prototype(feature,mask):  #return B*C*feature_size

    # print(feature.shape) #torch.Size([24, 32, 256, 256])
    # print(mask.shape) #torch.Size([24, 4, 256, 256])

    batch_pro = torch.zeros(mask.shape[0], mask.shape[1], feature.shape[1]) #torch.Size([24, 4, 32])

    # print(batch_pro.shape)
    for i in range(mask.shape[1]):
        #遍历每个类别信息
        classmask = mask[:,i,:,:] #torch.Size([24, 256, 256])
        #求取每个类原型
        proclass = masked_average_pooling(feature,classmask.unsqueeze(1)) #torch.Size([24, 32])

        batch_pro[:,i,:] = proclass

    return batch_pro


#输入特征与原型
def similarity_calulation(feature,batchpro): #feature_size = B*C*H*W  batchpro= B*C*dim
    ##torch.Size([24, 32, 256, 256],torch.Size([24, 4, 32]

    B = feature.size(0) #32
    feature = feature.view(feature.size(0), feature.size(1), -1)  # [N, C, HW] [24,32,65536]
    feature = feature.transpose(1, 2)  # [N, HW, C] [24,65536,32]
    feature = feature.contiguous().view(-1, feature.size(2)) #torch.Size([1572864, 32])

    C = batchpro.size(1) #4
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2)) #[96,32]
    feature = F.normalize(feature, p=2.0, dim=1) #对行数进行2范数(所有元素平方和的开方) #torch.Size([1572864, 32])
    batchpro = F.normalize(batchpro, p=2.0, dim=1).cuda()

    similarity = torch.mm(feature, batchpro.T) #矩阵相乘 #torch.Size([1572864, 96])

    similarity = similarity.reshape(-1, B, C) #torch.Size([1572864, 24, 4])

    similarity = similarity.reshape(B, -1, B, C) #torch.Size([24, 65536, 24, 4])

    return similarity


def selfsimilaritygen(similarity):
    # print(similarity.shape) #torch.Size([24, 65536, 24, 4])

    B = similarity.shape[0] #24
    mapsize = similarity.shape[1] #65536
    C = similarity.shape[3] #4

    selfsimilarity = torch.zeros(B,mapsize,C) #torch.Size([24, 65536, 4])

    for i in range(similarity.shape[2]):
        selfsimilarity[i,:,:] = similarity[i,:,i,:]

    return selfsimilarity.cuda()


def entropy_value(p, C):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=2) / \
        torch.tensor(np.log(C))#.cuda()
    return y1


def agreementmap(similarity_map):

    score_map = torch.argmax(similarity_map,dim=3) #torch.Size([24, 65536, 24]),类别预测图

    #score_map =score_map.transpose(1,2)
    ##print(score_map.shape, 'score',score_map[0,0,:])
    gt_onthot = F.one_hot(score_map,4) #torch.Size([18, 65536, 18, 4])

    # exit()
    avg_onehot = torch.sum(gt_onthot,dim=2).float() #torch.Size([18, 65536, 4])
    # print(avg_onehot.shape)

    avg_onehot = F.normalize(avg_onehot,1.0,dim=2) #torch.Size([18, 65536, 4])
    # print(avg_onehot.shape)

    # print(entropy_value(avg_onehot,similarity_map.shape[3]).shape) #torch.Size([18, 65536])


    ##print(gt_onthot[0,0,:,:],avg_onehot[0,0,:])
    weight = 1-entropy_value(avg_onehot,similarity_map.shape[3])
    ##print(weight[0,0])
    #score_map = torch.sum(score_map,dim=2)
    return weight


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            # nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class EmbeddingHead(nn.Module):
    def __init__(self, dim_in, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)

# torch.nn.functional.interpolate(dp3_out_seg, shape)

class DeepSupervision(nn.Module):
  def __init__(self, in_chn, out_chn, resize_to=[256,256]):
    super(DeepSupervision, self).__init__()
    self.dsv = nn.Sequential(
      nn.Conv2d(in_chn, out_chn, 1, 1, bias=True),
      # nn.Upsample(resize_to, mode='trilinear', align_corners=False)
      F.interpolate()
    )

  def forward(self, x):
    return self.dsv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)


        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

        self.embed_head = EmbeddingHead(dim_in=64, embed_dim=64)
        self.embed_head1 = EmbeddingHead(dim_in=16, embed_dim=16)


    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        shape = [256,256]

        out_dsv = []

        x11 = self.up1(x4, x3) #torch.Size([18, 128, 32, 32])
        dp3_out_seg = self.out_conv_dp3(x11)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)
        out_dsv.append(dp3_out_seg)
        # embedding_feature = self.embed_head(x)

        x22 = self.up2(x11, x2)#torch.Size([18, 64, 64, 64])
        dp2_out_seg = self.out_conv_dp2(x22)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)
        out_dsv.append(dp2_out_seg)
        # embedding_feature = self.embed_head(x22)

        x33 = self.up3(x22, x1) #torch.Size([18, 32, 128, 128])
        dp1_out_seg = self.out_conv_dp1(x33)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)
        out_dsv.append(dp1_out_seg)
        # embedding_feature = self.embed_head(x)


        x44 = self.up4(x33, x0) #torch.Size([18 ,16, 256, 256])
        # embedding_feature1 = self.embed_head1(x44)
        # print(x.shape)

        output = self.out_conv(x44)

        # mask = torch.softmax(output, dim=1)  # torch.Size([18, 4, 256, 256])
        #
        # # 求每个类原型
        # batch_pro = batch_prototype(x44, mask)  # torch.Size([18, 4, 16])
        #
        # # 特征图与原型的相似度(形成多个分割概率矩阵)
        # similarity_map = similarity_calulation(x44, batch_pro)  # torch.Size([18, 65536, 18, 4])
        #
        # # 相似熵图
        # entropy_weight = agreementmap(similarity_map)  # torch.Size([18, 65536])
        # # print(entropy_weight.shape)
        # # exit()
        #
        # self_simi_map = selfsimilaritygen(similarity_map)  # B*HW*C torch.Size([18, 65536, 4])

        # return output, embedding_feature, out_dsv, embedding_feature1,self_simi_map,entropy_weight,x11,x33,x44
        return output

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x



def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        # print(x4.shape) #torch.Size([24, 256, 16, 16])
        x = self.up1(x4, x3) #torch.Size([24, 128, 32, 32])

        # print(x.shape)
        if self.training:
            # print('1111')
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5)) #torch.Size([24, 4, 32, 32]
            # print(dp3_out_seg.shape)
        else:
            # print('22222')
            dp3_out_seg = self.out_conv_dp3(x)
            # print(dp3_out_seg.shape)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)
        # print(dp3_out_seg.shape)
        # exit()

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg

    
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  # 'feature_chns': [16, 32, 64, 128, 256],
                  'feature_chns': [32, 64, 128, 256, 512],
                  # 'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        # output1, embedding_feature, out_sdm, embedding_feature1,self_simi_map,entropy_weight,x11,x33,x44 = self.decoder1(feature)
        output= self.decoder1(feature)
        # return output1, embedding_feature, out_sdm, embedding_feature1,self_simi_map,entropy_weight,x11,x33,x44
        return output



class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


class MCNet2d_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        return output1, output2
    
class MCNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3

class MCNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        params4 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 3,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        self.decoder4 = Decoder(params4)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)    
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
