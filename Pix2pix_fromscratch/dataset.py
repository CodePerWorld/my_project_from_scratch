import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import config
import matplotlib.pyplot as plt

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_name = os.listdir(root_dir)
        print(self.images_name)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        image_name = self.images_name[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = np.array(Image.open(image_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]
        return input_image, target_image

    def get_test(self, index):
        return self.__getitem__(index)

def test():
    my_dataset = MyDataset("data/maps/train")
    input_image, target_image = my_dataset.get_test(0)
    a = torch.tensor([1,2,3,4])
    input_image, target_image = input_image.permute(1,2,0).numpy(), target_image.permute(1,2,0).numpy()
    print(type(input_image))
    _, axes = plt.subplots(1,2)
    axes[0].imshow(input_image)
    axes[1].imshow(target_image)
    plt.show()
if __name__ == '__main__':
    test()

