# Authors: rafik gouiaa <rafikgouiaaphd@gmail.com>, ...
import torch.nn as nn


class AlexNet(nn.Module):
    f"""
    This class implement AlexNet deep learning model
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.lrn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.lrn2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc8(x)
        x = self.softmax(x)
        return x
