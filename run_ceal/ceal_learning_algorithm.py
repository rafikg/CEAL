import numpy as np
from utils import Caltech256Dataset, Normalize, RandomCrop, SquarifyImage, \
    ToTensor
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import torch


def ceal_learning_algorithm(du: np.ndarray, dl: np.ndarray, k: int,
                            delta: float, dr: float, t: int, max_iter: int):
    """
    Algorithm1 : Learning algorithm of CEAL.
    For simplicity, I used the same notation in the paper.
    Parameters
    ----------
    du: np.ndarray
        Unlabeled samples
    dl : np.ndarray:
        labeled samples
    k: int, (default = 1000)
        uncertain samples selection
    delta: float
        hight confidence samples selection threshold
    dr: float
        threshold decay
    t: int
        fine-tuning interval
    max_iter: int
        maximum iteration number.

    Returns
    -------

    """
    pass


dataset = Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                            transform=transforms.Compose(
                                [SquarifyImage(),
                                 RandomCrop(224),
                                 Normalize(),
                                 ToTensor()]))

# Creating data indices for training and validation splits:
random_seed = 123
validation_split = .2
shuffling_dataset = True
batch_size = 16
dataset_size = len(dataset)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffling_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

model = AlexNet(n_classes=256, device=None)

model.train(epochs=50, train_loader=train_loader,
            valid_loader=validation_loader)

