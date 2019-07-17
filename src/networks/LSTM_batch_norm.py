import re

import torch
import torch.nn as nn

#torch.manual_seed(1)

from .LSTM import LanguageModel

#Network Definition
class BatchLanguageModel(LanguageModel):

    def __init__(self, cfg):
        super(BatchLanguageModel, self).__init__(cfg)

        self.txt_layernorm1 = nn.LayerNorm(self.hidden_dim)
        self.txt_layernorm2 = nn.LayerNorm(self.hidden_dim)

        self.to(self.device)

    def forward(self, ref=None):
        sentence = ref['vocab_tensor'][:, :-1]
        embeds = self.embedding(sentence)
        embeds = self.dropout1(embeds)
        n, m, b = embeds.size()

        embeds = self.txt_layernorm1(embeds)

        if 'feats' in ref:
            feats = ref['feats'].repeat(m, 1, 1).permute(1, 0, 2)

            #Concatenate text embedding and additional features
            #TODO fix for Maoetal_Full
            if embeds.size()[0]==1:
                embeds = torch.cat([embeds.repeat(feats.size()[0], 1, 1), feats], 2)
            else:
                embeds = torch.cat([embeds, feats], 2)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout2(lstm_out)
        lstm_out = self.txt_layernorm2(lstm_out)
        vocab_space = self.hidden2vocab(lstm_out)
        return vocab_space
