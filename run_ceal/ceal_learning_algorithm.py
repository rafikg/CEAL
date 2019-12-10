from utils import Caltech256Dataset, Normalize, RandomCrop, SquarifyImage, \
    ToTensor
from utils import get_uncertain_samples, get_high_confidence_samples, \
    update_threshold
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import torch
import logging

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def ceal_learning_algorithm(du: DataLoader,
                            dl: DataLoader,
                            dtest: DataLoader,
                            k: int = 1000,
                            delta_0: float = 0.005,
                            dr: float = 0.00033,
                            t: int = 1,
                            epochs: int = 10,
                            criteria: str = 'cl',
                            max_iter: int = 45):
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
    epochs: int
    criteria: str
    max_iter: int
        maximum iteration number.

    Returns
    -------

    """
    logger.info('Initial configuration: len(du): {}, len(dl): {} '.format(
        len(du.sampler.indices),
        len(dl.sampler.indices)))
    # Create the model
    model = AlexNet(n_classes=256, device=None)

    # Initialize the model
    logger.info('Intialize training the model on dl and test on dtest')
    model.train(epochs=epochs, train_loader=dl, valid_loader=None)

    # Evaluate model on dtest
    acc = model.evaluate(test_loader=dtest)
    print('====>Initial accuracy: {} '.format(acc))

    # High confidence samples
    for iteration in range(max_iter):
        logger.info('Calculate prediction on the unlabeled dataset `du`')
        pred_prob = model.predict(test_loader=du)

        # get k uncertain samples
        uncert_samp_idx = get_uncertain_samples(pred_prob=pred_prob, k=k,
                                                criteria=criteria)[:, 0]

        uncert_samp_idx = uncert_samp_idx.astype(int)

        uncert_samp_idx = [du.sampler.indices[idx] for idx in uncert_samp_idx]

        # add the uncertain samples selected from `du` to the labeled samples
        #  set `dl`
        dl.sampler.indices.extend(uncert_samp_idx)

        logger.info(
            'Update size of `dl`  and `du` by adding uncertain samples in `dl`'
            'and remove them from `du`'
            ' len(dl): {}, len(du) {}'.
                format(len(dl.sampler.indices), len(du.sampler.indices)))

        # Get high confidence samples `dh`
        hcs = get_high_confidence_samples(pred_prob=pred_prob, delta=delta_0)

        hcs_idx, hcs_labels = hcs[:, 0].astype(int), hcs[:, 1].astype(int)

        hcs_idx = [du.sampler.indices[idx] for idx in hcs_idx]

        # remove the samples that already selected as uncertain samples.
        hcs_idx = [x for x in hcs_idx if
                   x not in list(set(uncert_samp_idx) & set(hcs_idx))]

        print(len(hcs_idx))

        # add high confidence samples to the labeled set 'dl'
        dl.sampler.indices.extend(hcs_idx)  # update the indices
        for idx in range(len(hcs_idx)):
            dl.dataset.labels[hcs_idx[idx]] = hcs_labels[
                idx]  # update the original labels with the pseudo labels.

            if iteration % t == 0:
                logger.info('fine-tune the model on dh U dl')
                model.train(epochs=epochs, train_loader=dl)

                # update delta_0
                delta_0 = update_threshold(delta=delta_0, dr=dr, t=iteration)

            # remove the uncertain samples from the original du
            [du.sampler.indices.remove(idx) for idx in uncert_samp_idx]

            acc = model.evaluate(test_loader=dtest)
            print(
                "Iteration: {}, len(dl): {}, len(du): {},"
                " len(dh) {}, acc: {} ".format(
                    iteration, len(dl.sampler.indices),
                    len(du.sampler.indices), len(hcs_idx), acc))


if __name__ == "__main__":

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

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=train_sampler, num_workers=4)
    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=valid_sampler, num_workers=4)
    dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                        num_workers=4)

    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest)
