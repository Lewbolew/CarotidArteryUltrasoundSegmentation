import torch
import torch.nn as nn
import torch.nn.functional as F

from unet import UNet
from shape_net import ShapeUNet
BN_EPS = 1e-4


class SH_UNet(nn.Module):

    def __init__(self, path_to_shape_net_weights, n_classes=4):
        super(SH_UNet, self).__init__()

        self.unet = UNet((3, 544,544))
        self.shapeUNet = ShapeUNet((4, 544,544))
        # self.shapeUNet.load_state_dict(torch.load(path_to_shape_net_weights))

    def forward(self, x, only_encode=False):
        if only_encode:
            _, encoded_mask = self.shapeUNet(x)
            return encoded_mask

        unet_prediction = self.unet(x)
        input_shape_data = unet_prediction.detach()
        shape_net_final_prediction, shape_net_encoded_prediction = self.shapeUNet(input_shape_data)
        return unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction

