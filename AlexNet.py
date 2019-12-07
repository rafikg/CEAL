# Authors: rafik gouiaa <rafikgouiaaphd@gmail.com>, ...
import torch.nn as nn
import torch.optim as optim
import torch

from torchvision.models import alexnet
from torch.utils.data import DataLoader


class AlexNet(object):
    """
    Encapsulate the pretrained alexnet model
    Parameters
    ----------
    n_classes : int, default(256)
        the new number of classes
    device: str 'cuda' or 'cpu', default('cuda')
    """
    def __init__(self, n_classes: int = 256, device: int = 'cuda'):

        self.n_classes = n_classes
        self.model = alexnet(pretrained=True, progress=True)

        self.__freeze_all_layers()
        self.__change_last_layer()

        self.device = device

    def __freeze_all_layers(self) -> None:
        """
        freeze all layers in alexnet
        Returns
        -------
        None
        """

        for param in self.model.parameters():
            param.requires_grad = False

    def __change_last_layer(self) -> None:
        """
        change last layer to accept n_classes instead of 1000 classes
        Returns
        -------
        None
        """
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)

    def __add_softmax(self) -> None:
        """
        Add softmax layer to alexnet model
        Returns
        -------

        """
        # add softmax layer
        self.model = nn.Sequential(self.model, nn.Softmax(dim=1))

    def __train_one_epoch(self, train_loader: DataLoader,
                          epoch: int = 0, each_batch_idx: int = 100) -> None:
        """
        Train alexnet for one epoch
        Parameters
        ----------
        train_loader : DataLoader
        epoch : int
        each_batch_idx : int
            print training stats after each_batch_idx
        Returns
        -------
        None
        """
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.train()

        train_loss = 0
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
            momentum=0.9)

        criterion = nn.CrossEntropyLoss()

        for batch_idx, sample_batched in enumerate(train_loader):
            # load data and label
            data, label = sample_batched['image'], sample_batched['label']

            # convert data and label to be compatible with the device
            data = data.to(self.device)
            data = data.float()
            label = label.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run forward
            pred_prob = self.model(data)

            # calculate loss
            loss = criterion(pred_prob, label)

            # calculate gradient (backprop)
            loss.backward()

            # total train loss
            train_loss += loss.item()

            # update weights
            optimizer.step()

            if batch_idx % each_batch_idx == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           each_batch_idx * batch_idx / len(train_loader),
                           loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                            train_loss / len(
                                                                train_loader.dataset)))

    def train(self, epochs: int, train_loader: DataLoader) -> None:
        """
        Train alexnet for several epochs
        Parameters
        ----------
        epochs : int
            number of epochs
        train_loader:  DataLoader
            training set

        Returns
        -------
        None
        """
        for epoch in range(epochs):
            self.__train_one_epoch(train_loader=train_loader, epoch=epoch)

    def test_alexnet(self, test_loader: DataLoader) -> float:
        """
        Calaculate alexnet accuracy on test data
        Parameters
        ----------
        test_loader: DataLoader

        Returns
        -------
        accuracy: float
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, labels = sample_batched['image'], \
                               sample_batched['label']
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the: %d %%' % (
                100 * correct / total))

# TODO
# Fix some bugs in the training process of AlexNet. It is too slow and
# and the loss is not stable