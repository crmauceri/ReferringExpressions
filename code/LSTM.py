import os.path, argparse, re, sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

from ClassifierHelper import Classifier
from ReferExpressionDataset import ReferExpressionDataset

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, checkpt_file=None, vocab=None, hidden_dim=None, dropout=0,
            use_cuda=False, additional_feat=0):
        super(LanguageModel, self).__init__(use_cuda)

        if checkpt_file is not None:
            m = re.search('hidden(?P<hidden>\d+)_feats(?P<feats>\d+)_dropout(?P<dropout>\d+)', checkpt_file)
            self.hidden_dim = int(m.group('hidden'))
            self.feats_dim = int(m.group('feats'))
            self.dropout_p = float(m.group('dropout'))
        else:
            self.hidden_dim = hidden_dim
            self.dropout_p = dropout
            self.embed_dim = hidden_dim
            self.feats_dim = additional_feat

        #Word Embeddings
        self.word2idx = dict(zip(vocab, range(len(vocab))))
        self.ind2word = vocab
        self.vocab_dim = len(vocab)
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.embed_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + self.feats_dim, self.hidden_dim, dropout=self.dropout_p)
        self.hidden2vocab = nn.Linear(self.hidden_dim, self.vocab_dim)
        self.hidden = self.init_hidden()

        self.to(self.device)
        if checkpt_file is not None:
            super(LanguageModel, self).load_model(checkpt_file)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim, device=self.device, requires_grad=True),
                torch.zeros(1, 1, self.hidden_dim, device=self.device, requires_grad=True))

    @staticmethod
    def get_checkpt_file(checkpt_file, hidden_dim, feats_dim, dropout_p):
        return '{}_hidden{}_feats{}_dropout{}.mdl'.format(checkpt_file, hidden_dim, feats_dim, dropout_p)

    def checkpt_file(self, checkpt_file):
        return self.get_checkpt_file(checkpt_file, self.hidden_dim, self.feats_dim, self.dropout_p)

    def forward(self, ref=None, parameters=None):
        sentence = ref['vocab_tensor'][:-1]
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds)
        n, m = embeds.size()

        if 'feats' in ref:
            feats = ref['feats'].repeat(n, 1)

            #Concatenate text embedding and additional features
            embeds = torch.cat([embeds, feats], 1)

        lstm_out, self.hidden = self.lstm(embeds.view(n, 1, -1), self.hidden)
        vocab_space = self.hidden2vocab(lstm_out.view(len(sentence), -1))
        vocab_scores = F.log_softmax(vocab_space, dim=1)
        return vocab_scores

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def targets(self, ref):
        return torch.tensor(ref['vocab_tensor'][1:], dtype=torch.long, requires_grad=False, device=self.device)

    def clear_gradients(self):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden()

    def generate(self, start_word='<bos>', feats=None):
        sentence = []
        word_idx = self.word2idx[start_word]
        end_idx = self.word2idx['<eos>']

        with torch.no_grad():
            self.init_hidden()

            while word_idx != end_idx and len(sentence) < 100:
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                word_idx = torch.argmax(output)
                sentence.append(self.ind2word[word_idx])

        return sentence


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test')
    parser.add_argument('checkpoint_prefix',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

    parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/')
    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
                        help='Size of LSTM embedding (Default:100)')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    with open('vocab_file.txt', 'r') as f:
        vocab = f.read().split()
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_cuda)

    checkpt_file = LanguageModel.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 0, args.dropout)
    if (os.path.isfile(checkpt_file)):
        model = LanguageModel(checkpt_file=checkpt_file, vocab=vocab, use_cuda=use_cuda)
    else:
        model = LanguageModel(vocab=vocab, hidden_dim=args.hidden_dim,
                                  use_cuda=use_cuda, dropout=args.dropout)

    if(args.mode == 'train'):
        print("Start Training")
        total_loss = model.train(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image':False})

    if(args.mode == 'test'):
        print("Start Testing")
        print(model.generate([]))
