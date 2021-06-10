import os
import numpy as np
import torch
from torch import nn
from torchvision import models
# from matplotlib import pyplot as plt



class TSN(nn.Module):

    def __init__(self, num_segments, num_classes, modality = "rgb"):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.num_classes = num_classes
        # get pretrained model from resnet. For random init set this False
        self.model = models.resnet18(pretrained=True)
        # when using pretrained for optical flow, take average of first layer of weights and
        # duplicate 10 times to match shape of input stack
        if modality == "optical_flow":
            self.average_first_layer()
        # make last layer as fully connected
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_inputs, self.num_classes)
        # add softmax
        self.sm = nn.Softmax(dim=1)

    def average_first_layer(self):
        # take average of first layer and copy 10 times
        self.model.conv1.weight.data = self.model.conv1.weight.mean(1).unsqueeze(1).repeat_interleave(10,1)
    def forward(self,input):
        outputs = []
        # make computational graph for each segment using for loop
        for i in range(self.num_segments):
            frame_batches = input.permute(1,0,2,3,4)
            outputs.append(self.sm(self.model.float()(frame_batches[0].float())))
        y = torch.stack(outputs).mean(0) # take average of all outputs
        return y


