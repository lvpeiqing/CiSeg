import torch.nn as nn
import sys

sys.path.append('..')
import torch
import torch.nn.functional as F
import math
from torch.nn import init

affine_par = True
import copy


class PPMModule(nn.ModuleList):
    def __init__(self, pool_sizes=[1, 3, 6, 8]):
        super(PPMModule, self).__init__()
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size)
                )
            )

    def forward(self, x):
        out = []
        b, c, _, _ = x.size()
        for index, module in enumerate(self):
            out.append(module(x))
        # print(out[0].shape) #torch.Size([2, 256, 1, 1])
        # print(out[1].shape) #torch.Size([2, 256, 3, 3])
        # print(out[2].shape) #torch.Size([2, 256, 6, 6])
        # print(out[3].shape) #torch.Size([2, 256, 8, 8])
        # 最后输出时将其合并
        return torch.cat([output.view(b, c, -1) for output in out], -1)


class APNBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, value_channels, pool_sizes=[1, 3, 6, 8]):
        super(APNBBlock, self).__init__()

        # Generally speaking, here, in_channels==out_channels and key_channels==value_channles
        self.in_channels = in_channels
        self.out_channles = out_channels
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.pool_sizes = pool_sizes

        self.Conv_Key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        # 这里Conv_Query 和 Conv_Key权重共享，也就是计算出来的query和key是等同的
        self.Conv_Query = self.Conv_Key

        self.Conv_Value = nn.Conv2d(self.in_channels, self.value_channels, 1)
        self.Conv_Out = nn.Conv2d(self.value_channels, self.out_channles, 1)
        nn.init.constant_(self.Conv_Out.weight, 0)
        nn.init.constant_(self.Conv_Out.bias, 0)
        self.ppm = PPMModule(pool_sizes=self.pool_sizes)

    def forward(self, x):
        b, _, h, w = x.size()

        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        value = self.ppm(self.Conv_Value(x)).permute(0, 2, 1)
        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        query = self.Conv_Query(x).view(b, self.key_channels, -1).permute(0, 2, 1)
        # key = [batch, key_channels, 110]  where 110 = sum([s*2 for s in pool_sizes]) 1 + 3*2 + 6*2 + 8*2
        key = self.ppm(self.Conv_Key(x))

        # Concat_QK = [batch, h*w, 110]
        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *x.size()[2:])
        # Conv out
        Aggregate_QKV = self.Conv_Out(Aggregate_QKV)

        return Aggregate_QKV


class NonLocalNd_bn(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature,
                 with_gc, with_unary):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(NonLocalNd_bn, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.with_unary = with_unary

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        if 'channel' in self.whiten_type:
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type:
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        if self.with_unary:
            if query_mean.shape[1] == 1:
                query_mean = query_mean.expand(-1, key.shape[1], -1)
            unary = torch.bmm(query_mean.transpose(1, 2), key)
            unary = self.softmax(unary)
            out_unary = torch.bmm(value, unary.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_unary

        # out = residual + out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):

        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        # self.APNB = APNBBlock(in_channels=64, out_channels=64, value_channels=32, key_channels=32)
        # self.NonLocalNd_bn = NonLocalNd_bn(2,64,64   // 2,
        #                    downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False,
        #                    whiten_type=['channel'], temperature=1.0, with_gc=False, with_unary=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # if self.multi_level:
        self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.layer6 = ClassifierModule(2048+2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer7 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    def forward(self, x):

        # print(x.shape) #torch.Size([4, 3, 256, 256]),torch.Size([1, 3, 128, 128])
        x = self.conv1(x)  # torch.Size([4, 64, 128, 128]),torch.Size([1, 64, 64, 64])
        x = self.bn1(x)
        x = self.relu(x)

        # weight = self.APNB(x)
        # # weight = self.NonLocalNd_bn(x)
        # x = weight * x
        # # x_style = (1 - weight) * x

        x = self.maxpool(x)  # torch.Size([4, 64, 65, 65]),torch.Size([1, 64, 33, 33])
        x = self.layer1(x)  # torch.Size([4, 256, 65, 65]),torch.Size([1, 256, 33, 33])
        x = self.layer2(x)  # torch.Size([4, 512, 33, 33]),torch.Size([1, 512, 17, 17])
        # x_1 = x
        # x_1 = self.layer2(x)  # torch.Size([4, 512, 33, 33]),torch.Size([1, 512, 17, 17])
        x = self.layer3(x)  # torch.Size([4, 1024, 33, 33]),torch.Size([1, 1024, 17, 17])
        # x_2 = x
        # x_2 = self.layer3(x_1)  # torch.Size([4, 1024, 33, 33]),torch.Size([1, 1024, 17, 17])


        # x1 = self.layer5(x)  # produce segmap 1 # inchannel = 1024, out_channel = num_classes torch.Size([4, 5, 33, 33]),torch.Size([1, 5, 17, 17]

        x2 = self.layer4(x)  # inchannel= 1024, out_channel = 2048  x2 class feature torch.Size([4, 2048, 33, 33]),torch.Size([1, 2048, 17, 17])最后一层特征图
        # x2 = self.layer4(x_2)  # inchannel= 1024, out_channel = 2048  x2 class feature torch.Size([4, 2048, 33, 33]),torch.Size([1, 2048, 17, 17])最后一层特征图

        # x3 = self.layer6(x2)  # produce segmap 2 inchannel_2048 out_channel = num_classes torch.Size([4, 5, 33, 33]) torch.Size([1, 5, 17, 17])


        # return x2, x1, x3
        return x, x2
        # return x_1, x_2,x2


    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k


    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        # if self.multi_level:
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def get_deeplab_v2_ours(num_classes=19):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

