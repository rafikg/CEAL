from utils import (Caltech256Dataset, Normalize, RandomCrop, SquarifyImage,
                   ToTensor)
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms

data_set = Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                             transform=transforms.Compose(
                                 [SquarifyImage(),
                                  RandomCrop(224),
                                  Normalize,
                                  ToTensor()]))
train_loader = DataLoader(data_set, batch_size=4,
                          shuffle=True, num_workers=4)

model = AlexNet(n_classes=256, device='cuda')

model.train(epochs=10, train_loader=train_loader)
