import torch
import torch.nn as nn
from torchvision import models

class TruncatedResNet(nn.Module):
    def __init__(self, resnet):
        super(TruncatedResNet, self).__init__()
        self.ResNet = resnet
        self.output_dim = 4096

    #Forward pass ignores average pooling and fully connected layers
    def forward(self, x):
        x = self.ResNet.conv1(x)
        x = self.ResNet.bn1(x)
        x = self.ResNet.relu(x)
        x = self.ResNet.maxpool(x)

        x = self.ResNet.layer1(x)
        x = self.ResNet.layer2(x)
        x = self.ResNet.layer3(x)
        x = self.ResNet.layer4(x)

        return x

class TruncatedVGGorAlex(nn.Module):
    def __init__(self, vgg, maxpool=False):
        super(TruncatedVGGorAlex, self).__init__()
        self.VGG = vgg
        #Remove last pooling layer
        if not maxpool:
            self.VGG.features = nn.Sequential(*list(vgg.features.children())[:-1])
            self.output_dim = (512, 14, 14)
        else:
            self.output_dim = (512, 7, 7)

     # Forward pass ignores classification layers
    def forward(self, x):
        return self.VGG.features(x)
