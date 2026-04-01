import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import io, transform
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import cv2


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
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # print(image.shape)
        # x, y = image.shape[0], image.shape[1]
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if image.ndim == 2:
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif image.ndim == 3:
            x, y, c = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        else:
            raise ValueError(f"Unsupported image ndim: {image.ndim}")
        # if label.ndim == 2:
        #     x, y = label.shape
        #     if x != self.output_size[0] or y != self.output_size[1]:
        #         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #         low_res_label = zoom(label, (self.low_res[0] / x, self.low_res[1] / y), order=0)
        # elif label.ndim == 3:
        #     x, y, c = label.shape
        #     if x != self.output_size[0] or y != self.output_size[1]:
        #         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        #         low_res_label = zoom(label, (self.low_res[0] / x, self.low_res[1] / y, 1), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        image = torch.from_numpy(image.astype(np.float32))
        if image.ndim == 2:
            # Grayscale: add channel and repeat to get RGB
            image = image.unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        elif image.ndim == 3:
            # Already RGB: convert to (c,h,w)
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        print(image.shape)
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            # # print(slice_name)
            # # data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # # data = np.load(data_path)
            # # print(data['imgs'].shape)
            # # print(data['gts'].shape)
            # # image, label = data['imgs'], data['gts']
            # img_path = os.path.join(os.path.join(self.data_dir,'imgs'), slice_name+'.npy')
            # lbl_path = os.path.join(os.path.join(self.data_dir,'gts'), slice_name+'.npy')
            # image, label = np.load(img_path), np.load(lbl_path)
            
            img_path = os.path.join(self.data_dir, 'images', slice_name + '.png')  # Use PNG/JPG
            lbl_path = os.path.join(self.data_dir, 'masks', slice_name + '.png')   # Use PNG/JPG
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            #Pre-processing
            a_min, a_max = -125, 275
            b_min, b_max = 0.0, 1.0
            image = image.astype(np.float32)
            image = np.clip(image, a_min, a_max)
            assert a_max != a_min
            image = (image - a_min) / (a_max - a_min)
            print(f"Image shape: {image.shape}")
            H, W, D = image.shape
            if image.shape[-1] == 4:
                image = image[:, :, :3] 
            # image = np.transpose(image, (2, 1, 0))
        
            # label = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            # label = np.float32(label>0)
            # print(f"Label shape: {label.shape}")
            label = io.imread(lbl_path,as_gray=True)
            label= np.float32(label > 0)  # Convert to binary if needed
            # label = label.astype(np.float32)
            # label = np.expand_dims(label,0)
            # label = np.transpose(label, (2, 1, 0))
            if image is None or label is None:
                raise FileNotFoundError(f"Image or label not found: {img_path}, {lbl_path}")
            image = transform.resize(image, (512, 512), order=3, preserve_range=True, mode="constant", anti_aliasing=True)
            label = transform.resize(label, (512, 512), order=0, preserve_range=True, mode="constant", anti_aliasing=False)
            # label = np.uint8(label)
            # image = image / 255.0
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['imgs'][:], data['gts'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['name'] = self.sample_list[idx].strip('\n')
        return sample
