import torch.nn as nn
import torchvision

def get_channels():
    return [64, 128, 256, 512]


def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    # the residual building block that forms the foundation of resnet3D
    def __init__(self, in_channels, channels, stride=1, change_dim=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3x3(in_channels, channels, stride) # first conv1
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(channels, channels) # second conv2
        self.bn2 = nn.BatchNorm3d(channels)
        # if the input size is different from output size , change dim so that the values can be added
        self.change_dim = change_dim
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # if size of identity is not same as x they cannot be added
        # so we change size here
        if self.change_dim is not None:
            identity = self.change_dim(identity)
        # add input to the output
        # the key idea of Resnets
        x += identity
        x = self.relu(x)

        return x


class Resnet3D(nn.Module):

    def __init__(self, block, layers, block_in_channels, num_classes=25, pretrained=True):
        super(Resnet3D, self).__init__()

        self.in_channels = block_in_channels[0] # rgb so this is 3
        self.conv1 = nn.Conv3d(3, self.in_channels, kernel_size=(7, 7, 7),
                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False) # according to paper
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) # according to paper
        self.layer1 = self._make_layer(block, block_in_channels[0], layers[0], stride=1) # make layer 1
        self.layer2 = self._make_layer(block, block_in_channels[1], layers[1], stride=2) # make layer 2
        self.layer3 = self._make_layer(block, block_in_channels[2], layers[2], stride=2) # make layer 3
        self.layer4 = self._make_layer(block, block_in_channels[3], layers[3], stride=2) # make layer 4
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_in_channels[3], num_classes) # fully connected
        if pretrained == True: # if pretrained is true, take weights from 2D by changing shape
            inflated_params = []
            my_model_params = list(self.parameters()) # get current model param
            resnet18_params = list(torchvision.models.resnet18(pretrained=True).parameters()) # get resnet2D pretrained params
            # make 2D weights to 3D copying along one dimension and dividing by kernel size (according to paper)
            for i in range(len(my_model_params) - 2):
                if (resnet18_params[i].shape) != (my_model_params[i].shape):
                    kernel_size = resnet18_params[i].shape[-1]
                    inflated_weights = resnet18_params[i].unsqueeze(2).repeat_interleave(kernel_size, 2) / kernel_size
                else:
                    inflated_weights = resnet18_params[i]
                my_model_params[i].data = inflated_weights # assign inflated weights to current 3D model
                inflated_params.append(inflated_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        change_dim = None
        if stride != 1: # when stride is not 1 that also corresponds to different input and output dim
            change_dim = nn.Sequential(
                conv1x1x1(self.in_channels, channels, stride=stride),
                nn.BatchNorm3d(channels)) # do 1x1 conv and increase channels
        layers = list()
        layers.append(block(self.in_channels, channels, stride=stride, change_dim=change_dim)) # append first  block
        self.in_channels = channels
        for i in range(1, blocks): # append remaining blocks
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)



