from utils import Caltech256Dataset, SquarifyImage, RandomCrop, ToTensor, \
    Normalize
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt


def test_data_squarify():
    data_set = Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                                 transform=SquarifyImage())
    idx = np.random.randint(0, 2000)
    sample = data_set[idx]
    assert sample['image'].shape == (256, 256, 3)


def test_data_random_crop():
    data_set = Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                                 transform=transforms.Compose(
                                     [SquarifyImage(), RandomCrop(224)]))
    idx = np.random.randint(0, 2000)
    sample = data_set[idx]
    assert sample['image'].shape == (224, 224, 3)
    plt.imshow(sample['image'])
    plt.show()


def test_data_to_tensor():
    data_set = Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                                 transform=transforms.Compose(
                                     [SquarifyImage(), RandomCrop(224),
                                      Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225]),
                                      ToTensor()]))
    idx = np.random.randint(0, 2000)
    sample = data_set[idx]
    assert isinstance(sample['image'], torch.Tensor)
    assert sample['image'].dtype == torch.float64
