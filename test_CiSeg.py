import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory,config
from PIL import Image
from config import get_config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from networks.deeplabv2_ours import get_deeplab_v2_ours
from utils.visual import *
import cv2
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/cardiac', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CiSeg', help='experiment_name')
parser.add_argument('--target_modality', type=str, default='CT', help='target_modality_name')
parser.add_argument('--model', type=str, default='deeplabv2_ours', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5, help='output channel of network')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def test_single_volume(case, net_l, net_b,test_save_path, FLAGS):

    if FLAGS.target_modality == "CT":
        save_img_path = test_save_path + "/img/" + str(case)
        save_gt_path = test_save_path +  "/gt/" + str(case)
        save_pred_path = test_save_path +  "/pred/" + str(case)
        save_ent_path = test_save_path + "/ent/" + str(case)
    elif FLAGS.target_modality == "MR":
        save_img_path = test_save_path +  "/img/" + str(case)
        save_gt_path = test_save_path +  "/gt/" + str(case)
        save_pred_path = test_save_path + "/pred/" + str(case)
        save_ent_path = test_save_path +  "/ent/" + str(case)

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    if not os.path.exists(save_ent_path):
        os.makedirs(save_ent_path)

    i = 0
    print(case)

    interp = nn.Upsample(size=(256, 256), mode='bilinear',
                         align_corners=True)

    if FLAGS.target_modality == "CT":

       h5f = h5py.File(FLAGS.root_path + "/val_ct_slice/{}.h5".format(case), 'r')

    if FLAGS.target_modality == "MR":

       h5f = h5py.File(FLAGS.root_path + "/val_mr_slice/{}.h5".format(case), 'r') #

    image = h5f['image'][:]
    label = h5f['label'][:]

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        # x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (128 / x,128 / y), order=0) #(128, 128)
        # slice1 = zoom(slice, (128 / x,128 / y), order=0) #(128, 128)
        slice1 = slice

        slice = np.expand_dims(slice, -1)  # (128, 128, 1)
        slice = np.tile(slice, [1, 1, 3])  # h*w*3,(128, 128, 3)

        slice = np.transpose(slice, (2, 0, 1))  # (3, 128, 128)
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # torch.Size([1, 3, 128, 128])

        # slice1 =slice
        # slice1 = slice1.transpose(1,2,0)
        slice1*=255
        img_save = Image.fromarray(slice1.astype('uint8'))
        img_save.save(save_img_path + "/slice_{}.png".format(ind))

        label_slice = label[ind, :, :]
        gt_save = decode_seg_map_sequence(label_slice) * 255
        gt_save = Image.fromarray(np.uint8(gt_save))
        gt_save.save(save_gt_path + "/slice_{}.png".format(ind))

        net_l.eval()
        net_b.eval()
        with torch.no_grad():

            out1_l, out2_l = net_l(input)
            out1_b, out2_b = net_b(input)

            out = torch.cat((out2_l, out2_b), dim=1)
            out = net_l.layer6(out)
            # out = net_b.layer6(out)
            out_main = interp(out)

            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy() #
            prediction[ind] = pred

            pred_trg = decode_seg_map_sequence(pred.copy()) * 255
            pred_trg = Image.fromarray(np.uint8(pred_trg))
            pred_trg.save(save_pred_path + "/slice_{}.png".format(ind))

            entropy_map = compute_entropy_map(out_main)
            entropy_map = entropy_map.cpu().data.numpy()

            entropy_map = normalize_ent(entropy_map)
            entropy_map = construct_color_img(entropy_map)
            cv2.imwrite(save_ent_path + "/slice_{}.png".format(ind), entropy_map)


    i = i + 1

    # 类别1,2,3,4
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    if np.sum(prediction == 4) == 0:
        fourth_metric = 0, 0, 0, 0
    else:
        fourth_metric = calculate_metric_percase(prediction == 4, label == 4)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return first_metric, second_metric, third_metric,fourth_metric


def Inference(FLAGS):

    if FLAGS.target_modality == "CT":

       with open(FLAGS.root_path + '/val_ct_slices.list', 'r') as f:
          image_list = f.readlines()

    if FLAGS.target_modality == "MR":

        with open(FLAGS.root_path + '/val_mr_slices.list', 'r') as f:
           image_list = f.readlines()

    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    print(image_list)

    snapshot_path = "./experiments/heart/mr2ct/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "./experiments/heart/mr2ct/{}/CiSeg_ct_{}_predictions/".format(FLAGS.exp, FLAGS.model)

    torch.cuda.synchronize()
    start = time.time()

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)

    os.makedirs(test_save_path)

    net_l = get_deeplab_v2_ours(num_classes=5).cuda()
    net_b = get_deeplab_v2_ours(num_classes=5).cuda()

    save_model_path_l = os.path.join(snapshot_path, 'model_l_{}_best_model.pth'.format(FLAGS.model))
    save_model_path_b = os.path.join(snapshot_path, 'model_b_{}_best_model.pth'.format(FLAGS.model))
    net_l.load_state_dict(torch.load(save_model_path_l), strict=False)
    net_b.load_state_dict(torch.load(save_model_path_b), strict=False)
    print("init weight from {}".format(save_model_path_l))
    print("init weight from {}".format(save_model_path_b))
    net_l.eval()
    net_b.eval()


    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    fourth_total = 0.0

    for case in tqdm(image_list):
        print(case)

        first_metric, second_metric, third_metric,fourth_metric = test_single_volume(case, net_l, net_b,test_save_path, FLAGS)

        print('####################################')
        print(first_metric)
        print(second_metric)
        print(third_metric)
        print(fourth_metric)
        print('####################################')

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        fourth_total += np.asarray(fourth_metric)

    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)


    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list), fourth_total / len(image_list)]


    return avg_metric, test_save_path


if __name__ == '__main__':

    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path= Inference(FLAGS)

    print('-----------------------')
    print("average/each category metric is ：{}".format(metric))
    print("average metric is {}".format((metric[0] + metric[1] + metric[2] + metric[3]) / 4))


    with open(test_save_path+'../performance_ct_CiSeg.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2]+ metric[3])/4))




