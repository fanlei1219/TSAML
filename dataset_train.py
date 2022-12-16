from os.path import splitext
import numpy as np
from os import listdir
from torch.utils.data import Dataset
import logging
from PIL import Image
import torch
from torchvision import transforms


class BasicDataset_train(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples!')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        mask_file = self.masks_dir + idx + '.png'
        img_file = self.imgs_dir + idx + '.png'

        image = Image.open(img_file)
        mask = Image.open(mask_file)


        tp = transforms.Compose(
            [
             transforms.Grayscale(),
             transforms.ToTensor()
             ]
        )
        tot = transforms.ToTensor()

        image = tot(image)
        mask = tot(mask)


        return {'image': image, 'mask': mask}