import torch
import torch.nn as nn
from torchvision import models

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list

from DepthAwareCNN.models.Deeplab import Deeplab_Solver

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

class TruncatedDepthAwareCNN(nn.Module):
    def __init__(self, depthAwareCNN):
        super(TruncatedDepthAwareCNN, self).__init__()
        self.DepthAwareCNN = depthAwareCNN

    def forward(self, x):
        return self.DepthAwareCNN.model.features(x)

    def freeze(self, fix_weights):
        child_counter = 0
        for child in self.DepthAwareCNN.model.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

class TruncatedMaskRCNN(nn.Module):
    def __init__(self, maskrcnn):
        super(TruncatedMaskRCNN, self).__init__()
        self.model = maskrcnn

    # Forward pass ignores classification layers
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        return self.model.backbone(images.tensors)

    def freeze(self, fix_weights):
        child_counter = 0
        for child in self.model.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1