import re

import torchvision.models as models
import torch.nn as nn

#torch.manual_seed(1)

from .AttentionModule import AttentionModule
from .TruncatedImageNetworks import VGGorAlex
from .LSTM import LanguageModel
from .ClassifierHelper import Classifier, SequenceLoss

# Inspired by "Stacked attention networks for image question answering."
# Yang, Zichao, et al.
# CVPR 2016
# Implemented by Cecilia Mauceri

#Network Definition
class SAN(Classifier):

    def __init__(self, cfg):
        super(SAN, self).__init__(loss_function = SequenceLoss(nn.CrossEntropyLoss()))

        #Text Embedding Network
        self.wordnet = LanguageModel(cfg)

        #Image Embedding Network
        self.imagenet = VGGorAlex(cfg, models.vgg16(pretrained=True))

        #Attention Module to merge text and image
        self.attend1 = AttentionModule(cfg)
        self.attend2 = AttentionModule(cfg)

        self.outLayer = nn.Sequential(nn.Linear(self.lstm_dim, self.label_dim), nn.LogSoftmax())

        self.to(self.device)


    def forward(self, instance):
        #Image features
        image = instance['image']
        if self.use_cuda:
            image = image.cuda()
        image_out = self.imagenet(image)

        #Text features
        lstm_out = self.wordnet(instance)

        #Attention Layers
        attend1_out, attend1_weights = self.attend1(lstm_out[-1, :, :].view(self.lstm_dim, -1),
                                                    image_out.view(self.lstm_dim, -1))
        attend2_out, attend2_weights = self.attend2(attend1_out, image_out.view(self.lstm_dim, -1))

        label_scores = self.outLayer(attend2_out.transpose(0, 1))
        return label_scores

    def trim_batch(self, instance):
        return self.wordnet.trim_batch(instance)

    def clear_gradients(self, batch_size):
        super(SAN, self).clear_gradients()
        self.wordnet.clear_gradients(batch_size)
