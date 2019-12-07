import torch
from dataset import Caltech256Dataset, SquarifyImage, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from AlexNet import train_alexnet_for_one_epoch

def test_dataset():
    transformed_dataset =Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories",
                                           transform=transforms.Compose([SquarifyImage(), ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'].size(), sample_batched['label'])
        if i_batch == 4:
            break


def test_train_alexnet():
    transformed_dataset =Caltech256Dataset(root_dir="../caltech256/256_ObjectCategories/",
                                           transform=transforms.Compose([SquarifyImage(), ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    for epoch in range(5):
        train_alexnet_for_one_epoch(train_loader=dataloader, epoch=epoch)


if __name__ == "__main__":
    test_train_alexnet()
