#!/usr/bin/python
# -*- coding: UTF-8 -*-
# from efficientnet_pytorch import EfficientNet
from efficientnet_torch.model import EfficientNet
import torch.nn as nn

class Model_close(nn.Module):
    def __init__(self, base_model, out_dim):
        super(Model_close, self).__init__()
        self.model_dict = {
            'E4': EfficientNet.from_pretrained('efficientnet-b4', num_classes=out_dim),
        }
        self.backbone = self.model_dict[base_model]

    def forward(self, inputs):
        x = self.backbone(inputs)
        return x
