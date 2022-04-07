# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch


class OpenPredicateClassifierNN(nn.Module):

    def __init__(self):
        super(OpenPredicateClassifierNN, self).__init__()

        # Get pretrained resnet backbone for visual features extraction
        pretrained_backbone = True
        trainable_backbone_layers = 5  # From PyTorch fasterrcnn_resnet50_fpn example
        self.backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)

        # First fully connected layer
        self.fc1 = nn.Linear(4096, 1)


    def forward(self, inputs):
        images = inputs

        # Get image features vector
        features = self.backbone(images)
        features = features['pool']
        features = torch.flatten(features, start_dim = 1)

        # Apply linear layer with relu activation
        x = self.fc1(features)

        return x
