from dataset import Caltech256Dataset, SquarifyImage, ToTensor, RandomCrop, \
    Normalize
from torchvision import transforms
from torch.utils.data import DataLoader
from AlexNet import AlexNet

transformed_dataset = Caltech256Dataset(
    root_dir="../caltech256/256_ObjectCategories/",
    transform=transforms.Compose(
        [SquarifyImage(), RandomCrop(target_size=224),
         Normalize(
             [0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225]),
         ToTensor()]))
model = AlexNet()
train_loader = DataLoader(transformed_dataset, batch_size=4,
                          shuffle=True, num_workers=4)

model.train(epochs=10, train_loader=train_loader)
