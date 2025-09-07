import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
from torchvision import transforms
from PIL import Image


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        if self.split == 'train':

            with open(self._base_dir + '/tr_ct_slices.list', 'r') as f1:

                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':

            with open(self._base_dir + '/val_ct_slices.list', 'r') as f:

                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            print(self.sample_list)

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":

            h5f = h5py.File(self._base_dir + "/tr_ct_slice/{}".format(case), 'r')

        else:

            h5f = h5py.File(self._base_dir + "/val_ct_slice/{}".format(case), 'r')

        image = h5f['image'][:] #(144, 144)
        label = h5f['label'][:] #(144, 144)

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class MRBaseDataSets(Dataset):
    def __init__(self, base_dir=None,  split='train', transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        if self.split == 'train':

            # with open(self._base_dir + '/tr_ct_slices.list', 'r') as f1:
            with open(self._base_dir + '/tr_mr_slices.list', 'r') as f1:

                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':

            # with open(self._base_dir + '/val_ct_slices.list', 'r') as f:
            with open(self._base_dir + '/val_mr_slices.list', 'r') as f:

                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            print(self.sample_list)
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if self.split == "train":

            h5f = h5py.File(self._base_dir + "/tr_mr_slice/{}".format(case), 'r')
            # h5f = h5py.File(self._base_dir + "/tr_ct_slice/{}".format(case), 'r')

        else:

            # h5f = h5py.File(self._base_dir + "/val_ct_slice/{}".format(case), 'r')
            h5f = h5py.File(self._base_dir + "/val_mr_slice/{}".format(case), 'r')

        image = h5f['image'][:]
        label = h5f['label'][:]

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        sample["case"] = case
        return sample

class CTBaseDataSets(Dataset):
    def   __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        if self.split == 'train':

            with open(self._base_dir + '/tr_ct_slices.list', 'r') as f1:
            # with open(self._base_dir + '/tr_mr_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':

            with open(self._base_dir + '/val_ct_slices.list', 'r') as f:
            # with open(self._base_dir + '/val_mr_slices.list', 'r') as f:

                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            print(self.sample_list)
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if self.split == "train":

            h5f = h5py.File(self._base_dir + "/tr_ct_slice/{}".format(case), 'r')
            # h5f = h5py.File(self._base_dir + "/tr_mr_slice/{}".format(case), 'r')

        else:

            h5f = h5py.File(self._base_dir + "/val_ct_slice/{}".format(case), 'r')
            # h5f = h5py.File(self._base_dir + "/val_mr_slice/{}".format(case), 'r')

        image = h5f['image'][:]
        label = h5f['label'][:]

        if self.split == "val":
            image = image.astype(np.float32)
            label = label.astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)

        sample["idx"] = idx
        sample["case"] = case
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape

        # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        img = np.expand_dims(image, -1)  # (128, 128, 1)
        img = np.tile(img, [1, 1, 3])  # h*w*3,(128, 128, 3)

        image = np.transpose(img, (2, 0, 1))
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample
