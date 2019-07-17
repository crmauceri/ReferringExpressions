import torch
import torch.nn as nn
import numpy as np
from .ClassifierHelper import Classifier


class ImageClassifier(Classifier):
    def __init__(self, cfg):
        super(ImageClassifier, self).__init__(cfg, loss_function=MultiplePredictionLoss(cfg, nn.CrossEntropyLoss()))

    def trim_batch(self, instance):
        return instance['image'], instance['class_tensor']


class TruncatedResNet(ImageClassifier):
    def __init__(self, cfg, resnet):
        super(TruncatedResNet, self).__init__(cfg)
        self.ResNet = resnet
        self.output_dim = 4096

        self.to(self.device)

    #Forward pass ignores average pooling and fully connected layers
    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()
        x = self.ResNet.conv1(x)
        x = self.ResNet.bn1(x)
        x = self.ResNet.relu(x)
        x = self.ResNet.maxpool(x)

        x = self.ResNet.layer1(x)
        x = self.ResNet.layer2(x)
        x = self.ResNet.layer3(x)
        x = self.ResNet.layer4(x)

        return x


class TruncatedVGGorAlex(ImageClassifier):
    def __init__(self, cfg, vgg): #, maxpool=False, ignore_classification=False, fix_weights=None, checkpoint=None):
        super(TruncatedVGGorAlex, self).__init__(cfg)
        self.VGG = vgg
        self.ignore_classification = cfg.IMG_NET.IGNORE_CLASSIFICATION

        #Remove last pooling layer
        if not cfg.IMG_NET.MAXPOOL:
            self.VGG.features = nn.Sequential(*list(vgg.features.children())[:-1])
            self.output_dim = (512, 14, 14)
        else:
            self.output_dim = (512, 7, 7)

        if cfg.IMG_NET.FIX_WEIGHTS is not None:
            self.freeze(cfg.IMG_NET.FIX_WEIGHTS)

        self.to(self.device)

     # Forward pass ignores classification layers
    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

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


class DepthVGGorAlex(TruncatedVGGorAlex):
    def __init__(self, cfg, vgg):
        super(DepthVGGorAlex, self).__init__(cfg, vgg)

        depth_input_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.VGG.features = nn.Sequential(depth_input_layer, *list(self.VGG.features.children())[1:])
        self.to(self.device)

class MultiplePredictionLoss(nn.Module):
    def __init__(self, cfg, loss_function):
        super(MultiplePredictionLoss, self).__init__()
        self.loss = loss_function

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets, per_instance=False):
        # Targets is an n-hot representing multiple correct class labels
        # Randomly select a target
        target_index = torch.tensor([np.random.choice(r.cpu().nonzero()[0]) for r in targets], device=self.device)

        # Mask other targets to prevent gradient propegation
        mask = targets.clone().detach()
        mask[:, target_index] = 0
        embeddings[mask==1] = 0

        loss = self.loss(embeddings, target_index)
        return loss