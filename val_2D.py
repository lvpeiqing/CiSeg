import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
        # return dice
    else:
        return 0, 0
        # return 0


def denorm3(img):
    # convert [-1,1] to [0,1]
    # to use torchvision.utils.save_image
    img = (img) / 127.5 - 1
    # print(np.max(image))
    # image = image.astype(np.uint8)
    return img

def denorm(img):


    # img =img.cpu().detach().numpy()

    min = np.min(img)
    max = np.max(img)
    img = (img - (min)) / (max - min) * 255
    img= img.astype(np.uint8)
    # print(np.max(img))

    # img = torch.from_numpy(img).float().cuda()

    return img


def test_single_volume(image, label, net, classes, patch_size=[256,256]):

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)


    interp = nn.Upsample(size=(256, 256), mode='bilinear',
                         align_corners=True)


    for ind in range(image.shape[0]):

        slice = image[ind, :, :] #(512, 512)

        # x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0) #(256, 256)
        slice = np.expand_dims(slice, -1)  # (256, 256, 1)
        slice = np.tile(slice, [1, 1, 3])  # h*w*3,(256, 256, 3)


        slice = np.transpose(slice, (2, 0, 1))
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            _,_,out = net(input)
            out = interp(out) #torch.Size([1, 5, 512, 512])
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy()
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def  test_single_volumeall(image, label, net_l, net_b, classes, patch_size=[256, 256]):

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    interp = nn.Upsample(size=(256, 256), mode='bilinear',
                         align_corners=True)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]

        # x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0) #
        slice = np.expand_dims(slice, -1)
        slice = np.tile(slice, [1, 1, 3])  # h*w*3

        slice = np.transpose(slice, (2, 0, 1))
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()

        net_l.eval()
        net_b.eval()
        with torch.no_grad():

            out1_l,out2_l = net_l(input)
            out1_b, out2_b = net_b(input)

            out = torch.cat((out2_l, out2_b), dim=1)
            out = net_l.layer6(out)

            out = interp(out)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy()
            # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred


    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



