import torch
import torch.nn as nn
import numpy as np
from .ClassifierHelper import Classifier
from sklearn.metrics import precision_score, recall_score, hamming_loss


class ImageClassifier(Classifier):
    def __init__(self, cfg, loss_function):
        super(ImageClassifier, self).__init__(cfg, loss_function)

    def trim_batch(self, instance):
        return instance['image'], torch.as_tensor(instance['class_tensor'], dtype=torch.float32)

    # Makes a dictionary of all the objects in an image and the classifier's confidence on them
    def test(self, instance, targets):
        with torch.no_grad():
            prediction = torch.nn.functional.sigmoid(self(instance))

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


class TruncatedVGGorAlex(ImageClassifier):
    def __init__(self, cfg, vgg, loss_function): #, maxpool=False, ignore_classification=False, fix_weights=None, checkpoint=None):
        super(TruncatedVGGorAlex, self).__init__(cfg, loss_function)
        self.VGG = vgg
        self.ignore_classification = cfg.IMG_NET.IGNORE_CLASSIFICATION

        #Change output dimention
        if cfg.IMG_NET.N_LABELS != 1000:
            vgg.classifier._modules['6'] = nn.Linear(4096, cfg.IMG_NET.N_LABELS)

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
    def __init__(self, cfg, vgg, loss_function):
        super(DepthVGGorAlex, self).__init__(cfg, vgg, loss_function)

        depth_input_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.VGG.features = nn.Sequential(depth_input_layer, *list(self.VGG.features.children())[1:])
        self.to(self.device)

# DEPRECIATED: Use BCEWithLogitsLoss instead
class MultiplePredictionLoss(nn.Module):
    def __init__(self, cfg, loss_function):
        super(MultiplePredictionLoss, self).__init__()

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.loss = loss_function.cuda()
        else:
            self.device = torch.device('cpu')
            self.loss = loss_function.cpu()

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