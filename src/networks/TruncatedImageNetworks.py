import torch
import torch.nn as nn
import numpy as np
from .ClassifierHelper import Classifier
from sklearn.metrics import hamming_loss


class ImageClassifier(Classifier):
    def __init__(self, cfg, loss_function):
        super(ImageClassifier, self).__init__(cfg, loss_function)

    def trim_batch(self, instance):
        return instance['image'], torch.as_tensor(instance['class_tensor'],
                                                  dtype=torch.float32, device=self.device)

    # Makes a dictionary of all the objects in an image and the classifier's confidence on them
    def test(self, instance, targets):
        with torch.no_grad():
            prediction = torch.sigmoid(self(instance))

        targets = targets == 1;
        prediction = prediction > 0.5;
        output = dict()
        output['Hamming_Loss'] = hamming_loss(targets, prediction)
        TP = np.logical_and(targets, prediction)
        output['TP_classes'] = [t[1].item() for t in torch.nonzero(TP)]
        FP = np.logical_and(np.logical_not(targets), prediction)
        output['FP_classes'] = [t[1].item() for t in torch.nonzero(FP)]
        FN = np.logical_and(targets, np.logical_not(prediction))
        output['FN_classes'] = [t[1].item() for t in torch.nonzero(FN)]

        return output


class TruncatedResNet(ImageClassifier):
    def __init__(self, cfg, resnet, loss_function):
        super(TruncatedResNet, self).__init__(cfg, loss_function)
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


# VGG and Alexnet have similar enough architectures that the same method can be used to freeze layers
class VGGorAlex(ImageClassifier):
    def __init__(self, cfg, vgg, loss_function):
        super(VGGorAlex, self).__init__(cfg, loss_function)
        self.VGG = vgg

        #Change output dimention
        if cfg.IMG_NET.N_LABELS != 1000:
            vgg.classifier._modules['6'] = nn.Linear(4096, cfg.IMG_NET.N_LABELS)

        if len(cfg.IMG_NET.FIX_WEIGHTS) > 0:
            self.freeze(cfg.IMG_NET.FIX_WEIGHTS)

        self.to(self.device)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        return self.VGG(x)

    # Remove gradients from network layers to freeze pretrained network
    def freeze(self, fix_weights):
        print("Freeze image network weights")
        child_counter = 0
        for child in self.VGG.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class DepthVGGorAlex(VGGorAlex):
    def __init__(self, cfg, vgg, loss_function):
        super(DepthVGGorAlex, self).__init__(cfg, vgg, loss_function)

        # Add a channel to the first layer of convolution to process RGB-D
        depth_input_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.VGG.features = nn.Sequential(depth_input_layer, *list(self.VGG.features.children())[1:])
        self.to(self.device)