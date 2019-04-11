import argparse, os, re

import torch
import torchvision.models as models
import torch.nn as nn

#torch.manual_seed(1)

from AttentionModule import AttentionModule
from TruncatedImageNetworks import TruncatedVGGorAlex
from LSTM import LanguageModel
from ClassifierHelper import Classifier
from ReferExpressionDataset import ReferExpressionDataset

# Inspired by "Stacked attention networks for image question answering."
# Yang, Zichao, et al.
# CVPR 2016
# Implemented by Cecilia Mauceri

#Network Definition
class SAN(Classifier):

    def __init__(self, checkpt_file=None, vocab=None, hidden_dim=None, dropout=0, use_cuda=False):
        super(SAN, self).__init__()

        if checkpt_file is not None:
            m = re.search('hidden(?P<hidden>\d+)_feats(?P<feats>\d+)_dropout(?P<dropout>\d+)', checkpt_file)
            self.hidden_dim = int(m.group('hidden'))
            self.feats_dim = int(m.group('feats'))
            self.dropout_p = float(m.group('dropout'))
        else:
            self.feats_dim = 2005
            self.hidden_dim = hidden_dim
            self.dropout_p = dropout

        #Text Embedding Network
        self.wordnet = LanguageModel(vocab=vocab, additional_feat=self.feats_dim, hidden_dim=self.hidden_dim,
                                     dropout=self.dropout_p, use_cuda=self.use_cuda)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(models.vgg16(pretrained=True), maxpool=True, fix_weights=range(40))

        #Attention Module to merge text and image
        self.attend1 = AttentionModule(k=self.hidden_dim, d=self.lstm_dim, m=self.image_dim[1]*self.image_dim[2], use_cuda=use_cuda)
        self.attend2 = AttentionModule(k=self.hidden_dim, d=self.lstm_dim, m=self.image_dim[1]*self.image_dim[2], use_cuda=use_cuda)

        self.outLayer = nn.Sequential(nn.Linear(self.lstm_dim, self.label_dim), nn.LogSoftmax())

        self.total_loss = []
        self.start_epoch = 0

        self.to(self.device)
        if checkpt_file is not None:
            super(SAN, self).load_model(checkpt_file)


    def forward(self, instance, parameters):
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

    @staticmethod
    def get_checkpt_file(checkpt_file, hidden_dim, feats_dim, dropout_p):
        return '{}_hidden{}_feats{}_dropout{:.1f}.mdl'.format(checkpt_file, hidden_dim, feats_dim, dropout_p)

    def checkpt_file(self, checkpt_prefix):
        return self.get_checkpt_file(checkpt_prefix, self.hidden_dim, self.feats_dim, self.dropout_p)


