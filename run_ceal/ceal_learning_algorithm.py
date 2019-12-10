import numpy as np
from utils import Caltech256Dataset, Normalize, RandomCrop, SquarifyImage, \
    ToTensor
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import torch


def ceal_learning_algorithm(du: DataLoader,
                            dl: DataLoader,
                            dtest: DataLoader,
                            k: int = 1000,
                            delta_0: float = 0.005,
                            dr: float = 0.00033,
                            t: int = 10,
                            max_iter: int = 100):
    """
    Algorithm1 : Learning algorithm of CEAL.
    For simplicity, I used the same notation in the paper.
    Parameters
    ----------
    du: DataLoader
        Unlabeled samples
    dl : DataLoader
        labeled samples
    dtest : DataLoader
        test data
    k: int, (default = 1000)
        uncertain samples selection
    delta_0: float
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
    # Create the model
    model = AlexNet(n_classes=256, device=None)

    # Initialize the model
    model.train(epochs=50, train_loader=dl,
                valid_loader=dtest)

    # High confidence samples
    for i in range(max_iter):
        pass



dataset_train = Caltech256Dataset(
    root_dir="../caltech256/256_ObjectCategories_train",
    transform=transforms.Compose(
        [SquarifyImage(),
         RandomCrop(224),
         Normalize(),
         ToTensor()]))

dataset_test = Caltech256Dataset(
    root_dir="../caltech256/256_ObjectCategories_test",
    transform=transforms.Compose(
        [SquarifyImage(),
         RandomCrop(224),
         Normalize(),
         ToTensor()]))

# Creating data indices for training and validation splits:
random_seed = 123
validation_split = 0.1  # 10%
shuffling_dataset = True
batch_size = 16
dataset_size = len(dataset_train)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffling_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
print(len(val_indices))
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                 sampler=train_sampler, num_workers=4)
dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                 sampler=valid_sampler, num_workers=4)
dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                    num_workers=4)

# model = AlexNet(n_classes=256, device=None)
#
# model.train(epochs=50, train_loader=dl,
#             valid_loader=None)

ceal_learning_algorithm(du=du, dl=dl, dtest=dtest)
