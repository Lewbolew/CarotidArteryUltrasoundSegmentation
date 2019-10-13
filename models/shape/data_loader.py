import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import cv2
from skimage import io
import random
from albumentations import (
    ShiftScaleRotate, Compose, IAAAdditiveGaussianNoise
)
from torchvision.utils import save_image
from torchvision.transforms import Normalize
random.seed(42)

class ShapeData(nn.Module):

    def __init__(self, root_dir, partition):
        self.root_dir = root_dir
        self.list_IDs = os.listdir(os.path.join(self.root_dir, 'x_{}'.format(partition)))
        self.partition = partition
        self.full_pack_augmentor = Compose([
                    ShiftScaleRotate(shift_limit=0.2, border_mode=cv2.BORDER_REPLICATE,p=0.4),
                    # IAAAdditiveGaussianNoise(p=0.5, scale=(0.0,150.0))
])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        img_path = os.path.join(self.root_dir, 'x_{}'.format(self.partition), self.list_IDs[index])
        mask_path = os.path.join(self.root_dir, 'y_{}'.format(self.partition), self.list_IDs[index])

        to_tensor = transforms.ToTensor()

        X = io.imread(img_path)
        cur_mean = np.mean(X, axis=0)
        cur_std = np.std(X, axis=0)

        X = to_tensor(X)
        y = io.imread(mask_path)
        y = torch.from_numpy(y).long()
        return X, y

