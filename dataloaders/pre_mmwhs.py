import os
import glob
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import itk
from scipy.ndimage import zoom


base_dir = './MMWHS'
save_dir = './MMWHS/premmwhs'


def read_nifti(path):
    itk_img = sitk.ReadImage(path)

    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr



def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def convert_labels(label):
    label[label==205] = 1
    label[label==420] = 2
    label[label==500] = 3
    label[label==820] = 4
    label[label>4] = 0
    return label


def read_reorient2RAI(path):
    itk_img = itk.imread(path)

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    itk_img = filter.GetOutput()

    itk_arr = itk.GetArrayViewFromImage(itk_img)
    return itk_arr


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



def process_npy():
    for tag in ['MR', 'CT']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, tag, f'imagesTr', '*.nii.gz'))):
            print(path)

            img_id = path.split('/')[-1].split('.')[0]
            print(img_id)

            img_ids.append(img_id)

            label_id= img_id[:-5] + 'label'

            image_path = os.path.join(base_dir,tag, f'imagesTr', f'{img_id}.nii.gz')
            label_path =os.path.join(base_dir,tag, f'labelsTr', f'{label_id}.nii.gz')

            if not os.path.exists(os.path.join(save_dir, 'processed')):
                os.makedirs(os.path.join(save_dir, 'processed'))

            image_arr = read_reorient2RAI(image_path)
            label_arr = read_reorient2RAI(label_path)

            image_arr = image_arr.astype(np.float32)
            label_arr = convert_labels(label_arr)


            if img_id == "mr_train_1002_image":
                label_arr[0:4, :, :] = 0
                label_arr[:, -10:-1, :] = 0
                label_arr[:, :, 0:4] = 0


            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
            d, h, w = image_arr.shape

            d_s = (d_s - 4).clip(min=0, max=d)
            d_e = (d_e + 4).clip(min=0, max=d)
            h_s = (h_s - 4).clip(min=0, max=h)
            h_e = (h_e + 4).clip(min=0, max=h)
            w_s = (w_s - 4).clip(min=0, max=w)
            w_e = (w_e + 4).clip(min=0, max=w)

            image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
            label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

            upper_bound_intensity_level = np.percentile(image_arr, 98)

            image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
            image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)
            print(np.max(image_arr),np.min(image_arr))

            image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())
            print(np.max(image_arr),np.min(image_arr))

            dn, hn, wn = image_arr.shape

            image_arr = zoom(image_arr, [1, 256 / hn, 256 / wn], order=0)
            label_arr = zoom(label_arr, [1, 256 / hn, 256 / wn], order=0)

            print(image_arr.shape, image_arr.shape)


            image = sitk.GetImageFromArray(image_arr)
            label = sitk.GetImageFromArray(label_arr)

            sitk.WriteImage(image, os.path.join(save_dir, 'processed', f'{img_id}.nii.gz'))
            sitk.WriteImage(label, os.path.join(save_dir, 'processed', f'{label_id}.nii.gz'))




if __name__ == '__main__':
    process_npy()

