import torch
import torch.autograd as autograd
import torch.nn as nn

#torch.manual_seed(1)

# As described in "Stacked Attention Networks for Image Question Answering"
# Z. Yang, X. He, J. Gao et al.
# CVPR 2015
# Implemented by Cecilia Mauceri

class AttentionModule(nn.Module):
    def __init__(self, cfg):
        super(AttentionModule, self).__init__()
        self.k = cfg.LSTM.HIDDEN
        self.d = cfg.LSTM.EMBED
        self.m = cfg.IMG_NET.FEATS

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            use_cuda = True
            dtype = torch.cuda.FloatTensor
        else:
            use_cuda = False
            dtype = torch.FloatTensor

        self.W_word = nn.Linear(self.d, self.k, 1)
        self.W_image = autograd.Variable(torch.randn(self.k, self.d).type(dtype),
                                         requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        if use_cuda:
            self.cuda()

    #Inputs:
    # input_word : autograd.Variable dx1 tensor
    # input_image : autograd.Variable dxm tensor
    #
    #Outputs:
    # output : 1xd tensor
    # hidden_weights : attention weights used for visualizing attention 1xm tensor
    def forward(self, input_word, input_image):
        weighted_word = self.W_word(input_word.transpose(0, 1)).transpose(0,1)
        weighted_image = torch.mm(self.W_image,input_image)
        hidden_layer = self.tanh(torch.add(weighted_image, weighted_word.expand_as(weighted_image)))
        hidden_weights = self.softmax(hidden_layer)

        weighted_image = torch.sum(torch.mul(hidden_weights, input_image), dim=1)
        output = torch.add(weighted_image.view(self.d, -1), input_word)

        return output, hidden_weights
