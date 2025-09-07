import glob
import os
from tqdm import tqdm
import h5py
import numpy as np
import SimpleITK as sitk

slice_num = 0

mask_path = sorted(glob.glob("./tr_ct/image/*.nii.gz"))


################tr data: nii to slice(h5)################

for case in tqdm(mask_path):

    print(case)

    img_itk = sitk.ReadImage(case)

    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    origin = img_itk.GetOrigin()

    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("image", "label").replace('img', 'label')
    # msk_path = case.replace("image", "label").replace('image', 'label')



    if os.path.exists(msk_path):

        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)

        print(image.shape)
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        print(item)

        if image.shape != mask.shape:
            print("Error")
        print(item)

        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                './{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1


################val data: nii to volume(h5)################

for item in tqdm(mask_path):

    name_image = str(item)
    print(name_image)

    name_image1 = name_image.split("/")[-1].split(".")[0]

    name_label = name_image.replace("image", "label").replace('img', 'label')
    # name_label = name_image.replace("image", "label").replace('image', 'label')

    print(name_label)

    itk_img = sitk.ReadImage(name_image)
    image = sitk.GetArrayFromImage(itk_img)
    print(image.shape)

    itk_label = sitk.ReadImage(name_label)
    label = sitk.GetArrayFromImage(itk_label)
    print(label.shape)
    print(np.max(label))
    print(np.min(label))


    assert(np.max(label) == 4 and np.min(label) == 0)
    assert(np.shape(label)==np.shape(image))

    f = h5py.File(('./val_ct_volume/'+name_image1 + '_norm.h5'), 'w')

    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()
