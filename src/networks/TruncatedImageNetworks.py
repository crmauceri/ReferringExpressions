import torch
import torch.nn as nn
import numpy as np
from .ClassifierHelper import Classifier
from sklearn.metrics import hamming_loss


class ImageClassifier(Classifier):
    """ Parent class for Image Classifiers handles input formatting and testing """

    def __init__(self, cfg, loss_function):
        super(ImageClassifier, self).__init__(cfg, loss_function)

    def trim_batch(self, instance):
        """ Formats image classifier input"""
        return instance['image'], torch.as_tensor(instance['class_tensor'],
                                                  dtype=torch.float32, device=self.device)

    def test(self, instance, targets):
        """ Makes a dictionary of all the objects in an image and the classifier's confidence on them """
        with torch.no_grad():
            prediction = torch.sigmoid(self(instance))

        targets = (targets == 1).cpu()
        prediction = (prediction > 0.5).cpu()
        output = dict()
        output['Hamming_Loss'] = hamming_loss(targets, prediction)
        TP = np.logical_and(targets, prediction)
        output['TP_classes'] = [t[1].item() for t in torch.nonzero(TP)]
        FP = np.logical_and(np.logical_not(targets), prediction)
        output['FP_classes'] = [t[1].item() for t in torch.nonzero(FP)]
        FN = np.logical_and(targets, np.logical_not(prediction))
        output['FN_classes'] = [t[1].item() for t in torch.nonzero(FN)]

        return output

    def run_metrics(self, output, coco_dataset):
        coco = coco_dataset.coco
        metric_dict = dict()
        hamming_loss = 0.0
        TP = np.zeros((self.cfg.IMG_NET.N_LABELS + 1,))
        FP = np.zeros((self.cfg.IMG_NET.N_LABELS + 1,))
        FN = np.zeros((self.cfg.IMG_NET.N_LABELS + 1,))
        total = 0.0

        # load generation outputs
        for row in output:
            total += 1.0
            hamming_loss += row['Hamming_Loss']
            TP[row['TP_classes']] += 1
            FP[row['FP_classes']] += 1
            FN[row['FN_classes']] += 1

        metric_dict['HammingLoss'] = hamming_loss / total
        metric_dict["Precision"] = np.sum(TP) / (np.sum(TP) + np.sum(FP))
        metric_dict["Recall"] = np.sum(TP) / (np.sum(TP) + np.sum(FN))

        metric_dict["PerClass"] = list()
        for idx in range(self.cfg.IMG_NET.N_LABELS):
            label = coco.cats[coco_dataset.coco_cat_map[idx]]
            metric_dict["PerClass"].append({'precision': TP[idx] / (TP[idx] + FP[idx]),
                                          'recall': TP[idx] / (TP[idx] + FN[idx]),
                                          'label': label['name']})

        return metric_dict


class TruncatedResNet(ImageClassifier):
    """ ResNet with average pooling and fully connected layers removed """

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


class VGGorAlex(ImageClassifier):
    """ VGG (or Alexnet) with optional frozen weights """

    def __init__(self, cfg, vgg, loss_function):
        super(VGGorAlex, self).__init__(cfg, loss_function)
        self.VGG = vgg

        #Change output dimension
        if cfg.IMG_NET.N_LABELS != 1000:
            vgg.classifier._modules['6'] = nn.Linear(4096, cfg.IMG_NET.N_LABELS)

        if len(cfg.IMG_NET.FIX_WEIGHTS) > 0:
            self.freeze(cfg.IMG_NET.FIX_WEIGHTS)

        self.to(self.device)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        return self.VGG(x)

    # VGG and Alexnet have similar enough architectures that the same method can be used to freeze layers
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
    """Standard VGG (or Alexnet) architecture modified to have a 4d convolution on the first layer to support RGB-D input"""

    def __init__(self, cfg, vgg, loss_function):
        super(DepthVGGorAlex, self).__init__(cfg, vgg, loss_function)

        # Add a channel to the first layer of convolution to process RGB-D
        depth_input_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.VGG.features = nn.Sequential(depth_input_layer, *list(self.VGG.features.children())[1:])
        self.to(self.device)
