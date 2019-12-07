# Authors: rafik gouiaa <rafikgouiaaphd@gmail.com>, ...
import torch.nn as nn
import torch.optim as optim

from torchvision.models import AlexNet, alexnet


def train_alexnet_for_one_epoch(train_loader, epoch=0, n_classes=256, device='cpu'):
    """
    Train alexnet
    Parameters
    ----------
    train_loader
    n_classes

    Returns
    -------

    """
    model = alexnet(pretrained=True, progress=True)
    print(model)
    model.classifier[6] = nn.Linear(4096, n_classes)

    # freeze layers as mentioned in the paper
    for param in model.parameters():
        if param is not model.classifier[6]:
            param.requires_grad = False

    model.to(device)
    model.train()
    train_loss = 0
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, sample_batched in enumerate(train_loader):
        data, label = sample_batched['image'], sample_batched['label']
        data.to(device)
        label.to(device)
        optimizer.zero_grad()

        # call your model
        pred_prob = model(data)
        loss = criterion(pred_prob, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return model

