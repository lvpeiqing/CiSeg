import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
# from metrics import dice_coef
# from metrics import dice
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.shape[1]
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)



def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot



class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:

            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='none', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):

        target = to_one_hot(target.to(torch.int64), 5)
        # target = to_one_hot(target.to(torch.int64), 2)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)

        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())




class SCELoss(nn.Module):
    def __init__(self, num_classes=5, a=0.1, b=1):
        super(SCELoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels[:].long()).to(self.device)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels.to(torch.int64), self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = label_one_hot.permute(0, 3, 2, 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)  #
        pred = torch.clamp(pred, min=1e-7, max=1.0)  #
        label_one_hot = F.one_hot(labels.to(torch.int64), self.num_classes).float().to(self.device)  #
        label_one_hot = label_one_hot.permute(0, 3, 2, 1)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        # return gce.mean()
        return gce


class DiceLossSDM(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossSDM, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        # self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        self.celoss = CrossEntropyLoss()

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label.unsqueeze(1))
        celoss = self.celoss(predict, label.long())
        loss = celoss + self.alpha * diceloss
        return loss


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


###############################################
# BCE = torch.nn.BCELoss()

def weighted_loss(pred, mask):
    BCE = torch.nn.BCELoss(reduction = 'none')
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask).float()
    wbce = BCE(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()  



def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2



def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
#     print('a',a.size())
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    return loss_diff_avg 



class PixelCLLossSrc(nn.Module):
    def __init__(self, cfg):
        super(PixelCLLossSrc, self).__init__()
        self.cfg = cfg

    def forward(self, semantic_prototype, src_feat, src_mask):
        """
        The proposed contrastive loss for class-wise alignment
        Args:
            semantic_prototype: (CxK) are source prototypes for K classes #torch.Size([2048, 19])
            src_feat: (BxCxHxW) are source feature map #torch.Size([1, 2048, 90, 160])
            src_mask: (BxHxW) are source mask #torch.Size([1, 90, 160])
            tgt_feat: (BxCxHxW) are target feature map #torch.Size([1, 2048, 64, 128])
            tgt_mask: (BxHxW) are target mask #torch.Size([1, 64, 128])

        Returns:
        """
        assert not semantic_prototype.requires_grad

        src_mask = F.interpolate(src_mask.unsqueeze(1).float(), (src_feat.size(2), src_feat.size(3)),
                                 mode='nearest').squeeze(1).long()

        # batch size, channel size, height and width of target sample
        B, C, Hs, Ws = src_feat.size()  #torch.Size([4, 2048, 17, 17])

        # number of class
        # K = semantic_prototype.size(1)
        semantic_prototype = F.normalize(semantic_prototype, p=2, dim=1)
        semantic_prototype = semantic_prototype.transpose(1, 0) #torch.Size([2048, 5])

        # reshape src_feat to (BxHsxWs, C)
        src_feat = F.normalize(src_feat, p=2, dim=1)  # channel wise normalize torch.Size([4, 2048, 17, 17])])

        src_feat = src_feat.transpose(1, 2).transpose(2, 3).contiguous() #torch.Size([4, 17, 17, 2048])
        src_feat = src_feat.view(-1, C) #torch.Size([1156, 2048])

        src_mask = src_mask.view(-1, ) #torch.Size([1156])
        src_dot_value = src_feat.mm(semantic_prototype) / 1.0   #torch.Size([1156, 5])

        cosine = src_dot_value.view(B, Hs, Ws,-1) #torch.Size([4, 17, 17, 5])
        cosine = cosine.transpose(2, 3).transpose(1, 2) #torch.Size([4, 5, 17, 17])


        ce_criterion = nn.CrossEntropyLoss()

        loss = ce_criterion(src_dot_value, src_mask)

        return loss,cosine



class PixelCLLossTrg(nn.Module):
    def   __init__(self, cfg):
        super(PixelCLLossTrg, self).__init__()
        self.cfg = cfg

    def forward(self, semantic_prototype, src_feat, src_mask):
        """
        The proposed contrastive loss for class-wise alignment
        Args:
            semantic_prototype: (CxK) are source prototypes for K classes #torch.Size([2048, 19])
            src_feat: (BxCxHxW) are source feature map #torch.Size([1, 2048, 90, 160])
            src_mask: (BxHxW) are source mask #torch.Size([1, 90, 160])
            tgt_feat: (BxCxHxW) are target feature map #torch.Size([1, 2048, 64, 128])
            tgt_mask: (BxHxW) are target mask #torch.Size([1, 64, 128])

        Returns:
        """

        # print('semantic_prototype.shape', semantic_prototype.shape)  # torch.Size([5, 2048])
        # print('src_feat.shape', src_feat.shape)  # torch.Size([4, 2048, 17, 17])
        # print('src_mask.shape,', src_mask.shape)  # torch.Size([4, 128, 128])

        semantic_prototype = semantic_prototype.detach()
        assert not semantic_prototype.requires_grad

        src_mask = F.interpolate(src_mask.unsqueeze(1).float(), (src_feat.size(2), src_feat.size(3)),
                                 mode='nearest').squeeze(1).long()



        # batch size, channel size, height and width of target sample
        B, C, Hs, Ws = src_feat.size()  #torch.Size([4, 2048, 17, 17])

        # number of class
        # K = semantic_prototype.size(1)
        semantic_prototype = F.normalize(semantic_prototype, p=2, dim=1)
        semantic_prototype = semantic_prototype.transpose(1, 0) #torch.Size([2048, 5])
        # semantic_prototype = F.normalize(semantic_prototype,p=2,dim=0) #torch.Size([2048, 5])
        # print(semantic_prototype)

        # reshape src_feat to (BxHsxWs, C)
        src_feat = F.normalize(src_feat, p=2, dim=1)  # channel wise normalize torch.Size([4, 2048, 17, 17])])

        src_feat = src_feat.transpose(1, 2).transpose(2, 3).contiguous() #torch.Size([4, 17, 17, 2048])
        src_feat = src_feat.view(-1, C) #torch.Size([1156, 2048])

        src_mask = src_mask.view(-1, ) #torch.Size([1156])

        src_dot_value = src_feat.mm(semantic_prototype) / 1.0  #torch.Size([1156, 5])
        cosine = src_dot_value.view(B, Hs, Ws, -1)  # torch.Size([4, 17, 17, 5])
        cosine = cosine.transpose(2, 3).transpose(1, 2)  # torch.Size([4, 5, 17, 17])


        ce_criterion = nn.CrossEntropyLoss()

        loss = ce_criterion(src_dot_value, src_mask)

        return loss,cosine
