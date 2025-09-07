import os
import glob
import numpy as np
from tqdm import tqdm
# from utils import read_list, read_nifti, config
import SimpleITK as sitk
import torch.nn.functional as F
import torch
from scipy.ndimage import zoom
import h5py


base_dir = './data/malbcv/Training'
save_dir = './data/malbcv/pre_malbcv'

def read_nifti(path):
    itk_img = sitk.ReadImage(path)

    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr,itk_img

def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')



def getRangeImageDepth(label):
    d = np.any(label, axis=(1,2))
    h = np.any(label, axis=(0,2))
    w = np.any(label, axis=(0,1))

    if len(np.where(d)[0]) >0:
        d_s, d_e = np.where(d)[0][[0,-1]]
    else:
        d_s = d_e = 0

    if len(np.where(h)[0]) >0:
        h_s,h_e = np.where(h)[0][[0,-1]]
    else:
        h_s = h_e = 0

    if len(np.where(w)[0]) >0:
        w_s,w_e = np.where(w)[0][[0,-1]]
    else:
        w_s = w_e = 0
    return d_s, d_e, h_s, h_e, w_s, w_e

def norm(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices


def img_and_label_resample(img_path, label_path, out_shape=[256, 256, 56]):

    img = sitk.ReadImage(img_path, sitk.sitkInt16)
    label = sitk.ReadImage(label_path,sitk.sitkInt8)


    original_spacing = img.GetSpacing()  # (x, y, z)
    original_size = img.GetSize()        # (x, y, z)


    spacing_x = original_spacing[0] * original_size[0] / out_shape[0]
    spacing_y = original_spacing[1] * original_size[1] / out_shape[1]
    spacing_z = original_spacing[2] * original_size[2] / out_shape[2]

    new_spacing = [spacing_x, spacing_y, spacing_z]
    new_size = [out_shape[0], out_shape[1], out_shape[2]]  # (x, y, z)


    img_resampler = sitk.ResampleImageFilter()
    img_resampler.SetInterpolator(sitk.sitkLinear)
    img_resampler.SetDefaultPixelValue(0)
    img_resampler.SetOutputSpacing(new_spacing)
    img_resampler.SetOutputOrigin(img.GetOrigin())
    img_resampler.SetOutputDirection(img.GetDirection())
    img_resampler.SetSize(new_size)
    img_resampled = img_resampler.Execute(img)

    label_resampler = sitk.ResampleImageFilter()
    label_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    label_resampler.SetDefaultPixelValue(0)
    label_resampler.SetOutputSpacing(new_spacing)
    label_resampler.SetOutputOrigin(label.GetOrigin())
    label_resampler.SetOutputDirection(label.GetDirection())
    label_resampler.SetSize(new_size)
    label_resampled = label_resampler.Execute(label)

    img_np = sitk.GetArrayFromImage(img_resampled)    # (z, y, x)
    label_np = sitk.GetArrayFromImage(label_resampled) # (z, y, x)
    return img_np, label_np




def process_npy():
    for tag in ['Tr']:
        print(tag)
        img_ids = [] #1_segmentation_T2SPIR.nii
        for path in tqdm(glob.glob(os.path.join(base_dir, f'images{tag}', '*.nii.gz'))):

            img_id = path.split('/')[-1].split('.')[0]
            print(img_id)

            img_ids.append(img_id)

            label_id = 'label' + img_id[3:]

            image_path = os.path.join(base_dir, f'images{tag}', f'{img_id}.nii.gz')
            label_path = os.path.join(base_dir, f'labels{tag}', f'{label_id}.nii.gz')

            image,label= img_and_label_resample(image_path,label_path, out_shape=[256, 256, 56])
            print(image.shape, label.shape)
            print('max0', np.max(image), 'min0', np.min(image))

            image[image > 275] = 275
            image[image < -125] = -125
            image = np.rot90(image)
            image = np.rot90(image)

            label = np.rot90(label)
            label = np.rot90(label)

            mask_data = np.zeros_like(label)

            mask_data[label == 6] = 1
            mask_data[label == 2] = 2
            mask_data[label == 3] = 3
            mask_data[label == 1] = 4
            print('max0', np.max(mask_data), 'min0', np.min(mask_data))

            image = image
            mask_data = mask_data.astype(np.int8)

            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(mask_data)
            d, h, w = image.shape

            d_s = (d_s - 2).clip(min=0, max=d)
            d_e = (d_e + 2).clip(min=0, max=d)
            h_s = (h_s - 30).clip(min=0, max=h)
            h_e = (h_e + 30).clip(min=0, max=h)
            w_s = (w_s - 30).clip(min=0, max=w)
            w_e = (w_e + 30).clip(min=0, max=w)

            image = image[d_s:d_e, h_s:h_e, w_s: w_e]
            label = mask_data[d_s:d_e, h_s:h_e, w_s: w_e]
            print(image.shape,label.shape)

            dn, hn, wn = image.shape
            image_padded = zoom(image, [1, 256 / hn, 256 / wn], order=1)
            label_padded = zoom(label, [1, 256 / hn, 256 / wn], order=0)

            print(image_padded.shape, label_padded.shape)

            image_padded = (image_padded - image_padded.mean()) / (image_padded.std() + 1e-8)
            print(np.min(image_padded), np.max(image_padded))

            image_padded = np.clip(image_padded, -3.0, 3.0)
            print(np.min(image_padded), np.max(image_padded))

            image_padded  = norm(image_padded)
            print(np.min(image_padded), np.max(image_padded))

            if not os.path.exists(os.path.join(save_dir, 'processed')):
                os.makedirs(os.path.join(save_dir, 'processed'))

            image_nii = sitk.GetImageFromArray(image_padded)
            label_nii = sitk.GetImageFromArray(label_padded)


            sitk.WriteImage(image_nii, os.path.join(save_dir, 'processed', f'{img_id}.nii.gz'))
            sitk.WriteImage(label_nii, os.path.join(save_dir, 'processed', f'{label_id}.nii.gz'))


if __name__ == '__main__':
    process_npy()
