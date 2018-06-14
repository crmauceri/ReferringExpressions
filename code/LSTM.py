import random, os, pickle, argparse
from tqdm import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from ClassifierHelper import Classifier

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, checkpt_file=None, word_embeddings=None, hidden_dim=None,
                 output_dim=None, use_cuda=False):
        super(LanguageModel, self).__init__(use_cuda)

        if checkpt_file is not None:
            super(LanguageModel, self).load_model(checkpt_file)
        else:
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.word_embeddings = word_embeddings

        embedding_dim = word_embeddings.weight.size()[1]

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        if self.use_cuda:
            self.cuda()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def load_params(self, checkpoint):
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']
        self.word_embeddings = checkpoint['word_embeddings']

    def save_model(self, checkpt_file, params):
        params['hidden_dim'] = self.hidden_dim
        params['output_dim'] = self.output_dim
        params['word_embeddings'] = self.word_embeddings
        super(LanguageModel, self).save_model(checkpt_file, params)

    def forward(self, sentence):
        if self.use_cuda:
            sentence = sentence.cuda()

        embeds = self.word_embeddings(sentence)
        out = self.lstm(embeds)
        return out

    def train(self, n_epochs, instances, checkpt_file):
        return super(LanguageModel, self).train(n_epochs, instances, checkpt_file)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test', default='train')
    parser.add_argument('checkpoint_file',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

    parser.add_argument('--data_root', help='path to data directory', default='data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')

    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=100,
                        help='Size of LSTM embedding (Default:100)')
    parser.add_argument('--use_tokens', dest='use_tokens', type=bool, default=True,
                        help='If false, ignores pos token features. (Default:True)')
    parser.add_argument('--tag_dim', dest='tag_dim', type=int, default=10,
                        help='Size of tag embedding. If <1, will use one-hot representation (Default:10)')

    args = parser.parse_args()
    if (args.checkpoint_file):
        args.data_root = 'data'

    use_cuda = torch.cuda.is_available()

    model = LanguageModel(word_embeddings, use_cuda=use_cuda)

    if(args.checkpoint_file):
        model.load_model(args.checkpoint_file)

    if(args.mode == 'train'):
        print("Start Training")
        total_loss = model.train(args.epochs, refer, args.checkpoint_file)

    if(args.mode == 'test'):
        print("Start Testing")
        model.test(refer)