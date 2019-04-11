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
    def __init__(self, vgg, maxpool=False, ignore_classification=False, fix_weights=None):
        super(TruncatedVGGorAlex, self).__init__()
        self.VGG = vgg
        self.ignore_classification = ignore_classification
        #Remove last pooling layer
        if not maxpool:
            self.VGG.features = nn.Sequential(*list(vgg.features.children())[:-1])
            self.output_dim = (512, 14, 14)
        else:
            self.output_dim = (512, 7, 7)

        if fix_weights is not None:
            self.freeze(fix_weights)

     # Forward pass ignores classification layers
    def forward(self, x):
        if self.ignore_classification:
            return self.VGG.features(x)
        else:
            return self.VGG(x)

    def freeze(self, fix_weights):
        child_counter = 0
        for child in self.VGG.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1
