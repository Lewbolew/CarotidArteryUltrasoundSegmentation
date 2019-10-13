import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import numpy as np
import os
import random
import cv2 as cv
from skimage import io

random.seed(42)

class UltrasoundData(nn.Module):
    
    def __init__(self, root_dir, partition):
        self.root_dir = root_dir
        self.list_IDs = os.listdir(os.path.join(self.root_dir, 'y_{}'.format(partition)))
        self.partition = partition

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        img_path = os.path.join(self.root_dir, 'x_{}'.format(self.partition), self.list_IDs[index])
        mask_path = os.path.join(self.root_dir, 'y_{}'.format(self.partition), self.list_IDs[index])

        X = io.imread(img_path)
        # X = np.load(img_path)
        y = io.imread(mask_path)
        # y = np.load(mask_path)
        
        to_tensor = transforms.ToTensor()
        X = to_tensor(X)
        # X = torch.from_numpy(X).unsqueeze(0).float()
        y = torch.from_numpy(y).long()
        return X, y

class UltrasoundDataShapeNet(UltrasoundData):

    def __init__(self, root_dir, partition):
        super().__init__(root_dir, partition)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        mask_path = os.path.join(self.root_dir, 'y_{}'.format(self.partition), self.list_IDs[index])

        y = io.imread(mask_path)
        
        X = (np.arange(4) == y[...,None]).astype(np.uint8)
        X[X==1] = 255
        X = self.corrupt_image(X, 'lumenTissue')
        to_tensor = transforms.ToTensor()
        X = to_tensor(X)

        y = torch.from_numpy(y).long()
        return X, y

    def corrupt_image(self, image, channels_corruption):
        def sp_noise(image,prob):
            '''
            Add salt and pepper noise to image
            prob: Probability of the noise
            '''
            output = np.zeros(image.shape,np.uint8)
            thres = 1 - prob 
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j] = 0
                    elif rdn > thres:
                        output[i][j] = float(random.randint(1, 255))
                    else:
                        output[i][j] = image[i][j]
            return output
        def corrupt_image(image, noise_intensity, num_of_iteratons, kernel):
            noise_img = sp_noise(image,noise_intensity)
            noise_img = cv.erode(noise_img, kernel, iterations=num_of_iteratons)
            return noise_img

        values_for_noise_intensity_lumen = [0.05, 0.1, 0.3, 0.8]
        values_for_noise_intensity_tissue = [0.005, 0.001, 0.003, 0.009]
        kernel_for_lumen = np.ones((4,4),np.uint8)
        kernel_for_tissue = np.ones((10,10), np.uint8)
        if channels_corruption == 'lumen':
            image[:,:, 2] = corrupt_image(image[:,:, 2], random.choice(values_for_noise_intensity_lumen), 1, kernel_for_lumen)
        elif channels_corruption == 'lumenTissue':
            image[:,:, 2] = corrupt_image(image[:,:, 2], random.choice(values_for_noise_intensity_lumen), 1, kernel_for_lumen)
            image[:,:,1] = corrupt_image(image[:,:,1], random.choice(values_for_noise_intensity_tissue), 
                                        1, kernel_for_tissue)


        return image
