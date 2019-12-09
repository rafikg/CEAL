# Authors: Rafik Gouiaa <rafikgouiaaphd@gmail.com>

# Modified from <https://pytorch.org/tutorials/beginner/
# data_loading_tutorial.html#transforms>

from typing import Optional, Callable, Union
from torch.utils.data import Dataset

import torch
import os
import glob
import numpy as np
import warnings
import cv2

warnings.filterwarnings("ignore")


class Caltech256Dataset(Dataset):
    """
    Encapsulate Caltech256 torch.utils.data.Dataset

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory.

    transform : Callable,
        A transform function that takes the original image and
        return a transformed version.

    Attributes
    ----------
    data : list
        list of images files names
    labels : list
        list of integers (labels)
    """

    def __init__(self, root_dir: str = "calthec256",
                 transform: Optional[Callable] = None):

        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.data = []
        self.labels = []
        self._classes = 256

        # load data and labels
        for cat in range(0, self._classes):
            cat_dir = glob.glob(
                os.path.join(self.root_dir, '%03d*' % (cat + 1)))[0]

            for img_file in glob.glob(os.path.join(cat_dir, '*.jpg')):
                self.data.append(img_file)
                self.labels.append(cat)

    def __getitem__(self, idx: int) -> dict:
        """
        Get the idx element

        Parameters
        ----------
        idx : int
           the index of the element


        Returns
        -------
        sample: dict[str, Any]
        """
        img, label = self.data[idx], self.labels[idx]
        img = cv2.imread(img)
        img = img[:, :, ::-1]
        img = self.img_normalize(img)
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):

        return len(self.data)

    @staticmethod
    def img_normalize(img):
        img = (img / 255.0)

        return img


class Normalize(object):
    """
    Normalize the image in the sample using
    imagenet parameters
    Parameters
    ----------
    mean: np.ndarray
        mean of imagenet training set
    std: np.ndarray
        std of image net training set
    """

    def __init__(self, mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
                 std: np.ndarray = np.array([0.229, 0.224, 0.225])):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img - self.mean
        img /= self.std
        sample = {'image': img, 'label': label}
        return sample


class SquarifyImage(object):
    """
    Scale and squarify an image into box of fixed ize

    Parameters
    ----------
    box_size :  int
        the size of the output box.
    scale : tuple
        min scale ratio and max scale ratio
    is_scale: bool
        flag to scale or not the image
    seed: Callable or int, optional

    """

    def __init__(self, box_size: int = 256, scale: tuple = (0.6, 1.2),
                 is_scale: bool = True,
                 seed: Optional[Union[Callable, int]] = None):
        super(SquarifyImage, self).__init__()
        self.box_size = box_size
        self.min_scale_ratio = scale[0]
        self.max_scale_ratio = scale[1]
        self.is_scale = is_scale
        self.seed = seed

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = self.squarify(img)
        sample = {'image': img, 'label': label}
        return sample

    def squarify(self, img):
        """
        Squarfiy the image
        Parameters
        ----------
        img : np.ndarray
            1-channel or 3-channels image

        Returns
        -------
        img_padded : np.ndarray
        """
        if self.is_scale:
            img_scaled = self.img_scale(img)
            img = img_scaled
        w, h, _ = img.shape

        ratio = min(self.box_size / w, self.box_size / h)
        resize_w, resize_h = int(w * ratio), int(h * ratio)
        x_pad, y_pad = (self.box_size - resize_w) // 2, (
                self.box_size - resize_h) // 2
        t_pad, b_pad = x_pad, self.box_size - resize_w - x_pad
        l_pad, r_pad = y_pad, self.box_size - resize_h - y_pad

        resized_img = cv2.resize(img, (resize_h, resize_w))

        img_padded = cv2.copyMakeBorder(resized_img,
                                        top=t_pad,
                                        bottom=b_pad,
                                        left=l_pad,
                                        right=r_pad,
                                        borderType=0,
                                        value=0)

        if img_padded.shape == [self.box_size, self.box_size, 3]:
            raise ValueError(
                'Invalid size for squarified image {} !'.format(
                    img_padded.shape))
        return img_padded

    def img_scale(self, img):
        """
        Randomly scaling an image
        Parameters
        ----------
        img  : np.ndarray
            1-channel or 3-channels image

        Returns
        -------
        img_scaled : np.ndarray
        """
        scale = np.random.uniform(self.min_scale_ratio, self.max_scale_ratio,
                                  self.seed)
        img_scaled = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img_scaled


class RandomCrop(object):
    """
    Randomly crop the image in the sample to a target size
    target_size: tuple(int, int) or int. If int, take a square crop.
        the desired crop size

    """

    def __init__(self, target_size: Union[tuple, int]):

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            assert len(target_size) == 2
            self.target_size = target_size

    def __call__(self, sample):

        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        new_h, new_w = self.target_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
              left: left + new_w]
        sample = {'image': img, 'label': label}
        return sample


class ToTensor(object):
    """
    Convert ndarrays image in sample to pytorch Tensor
    """

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img),
                  'label': label}
        return sample
