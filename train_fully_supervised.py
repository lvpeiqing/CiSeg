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

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils.losses import DiceLoss,DiceLossSDM
from val_2D import test_single_volume


cpu_num = 4 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default = './data/cardiac', help = 'Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='fully_supervised_ct_deeplabv2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='deeplabv2', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
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


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    # print(num_classes)
    # exit()
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    # model1 = ISFA(in_chns=3).cuda()

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            # print('-----------此时的训练的idex为-------,',sampled_batch["idx"])

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            # print(np.max(volume_batch.cpu().detach().numpy()))
            # print(np.min(volume_batch.cpu().detach().numpy()))
            # print(np.max(label_batch.cpu().detach().numpy()))
            # print(np.min(label_batch.cpu().detach().numpy()))
            # print(volume_batch.shape) #torch.Size([24, 3, 128, 128])
            # print(label_batch.shape) #torch.Size([24, 128, 128])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.subplot(121),plt.imshow(volume_batch[2, :,:, :].permute(1,2,0).cpu().detach().numpy(),cmap='gray')
            # plt.subplot(122),plt.imshow(label_batch.cpu().detach().numpy()[2, :, :])
            # plt.show()
            # exit()

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # outputs = model(volume_batch) #[24,5,128,128]
            _,pred_src_aux, pred_src_main = model(volume_batch) #torch.Size([24, 5, 17, 17]) torch.Size([24, 5, 33, 33])


            pred_src_aux = interp(pred_src_aux) #torch.Size([24, 5, 256, 256])
            pred_src_main = interp(pred_src_main)

            pred_src_main_soft = torch.softmax(pred_src_main, dim=1)
            pred_src_aux_soft = torch.softmax(pred_src_aux, dim=1)

            loss_ce_main = ce_loss(pred_src_main, label_batch[:].long())
            loss_dice_main = dice_loss(pred_src_main_soft, label_batch.unsqueeze(1))

            loss_ce_aux = ce_loss(pred_src_aux, label_batch[:].long())
            loss_dice_aux = dice_loss(pred_src_aux_soft, label_batch.unsqueeze(1))

            loss = 1.0 * (loss_ce_main + loss_dice_main) + 0.5 * (loss_ce_aux + loss_dice_aux)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_ce_aux: %f, loss_dice_aux: %f,' %
                (iter_num, loss.item(), loss_ce_main.item(), loss_dice_main.item(), loss_ce_aux.item(), loss_dice_aux.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):

                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)

                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                ##########################################
                print('metric_list,', metric_list)
                print('metric_list[MYO],', metric_list[0, 0])
                print('metric_list[LAC],', metric_list[1, 0])
                print('metric_list[LVC],', metric_list[2, 0])
                print('metric_list[AA],', metric_list[3, 0])
                #################################################

                # print('metric_list,', metric_list)
                # print('metric_list[Liver],', metric_list[0, 0])
                # print('metric_list[Lk],', metric_list[1, 0])
                # print('metric_list[Rk],', metric_list[2, 0])
                # print('metric_list[Spl],', metric_list[3, 0])
                ################################################


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
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
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



    snapshot_path = "./experiments/heart/ct2mr/{}/{}".format(args.exp, args.model)


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
