import os.path, argparse, re, sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

from ClassifierHelper import Classifier

from refer_python3.refer import REFER

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, checkpt_file=None, vocab=None, hidden_dim=None, dropout=0,
            use_cuda=False, additional_feat=0):
        super(LanguageModel, self).__init__(use_cuda)

        if checkpt_file is not None:
            m = re.match('hidden(?P<hidden>\d+)_vocab(?P<vocab>\d+)_embed(?P<embed>\d+)_feats(?P<feats>\d+)_dropout(?P<dropout>\d+)', checkpt_file)
            self.hidden_dim = m.group('hidden')
            self.vocab_dim = m.group('vocab')
            self.embed_dim = m.group('embed')
            self.feats_dim = m.group('feats')
            self.dropout_p = m.group('dropout')
        else:
            self.hidden_dim = hidden_dim
            self.dropout_p = dropout
            self.word2idx = dict(zip(vocab, range(len(vocab))))
            self.ind2word = vocab
            self.vocab_dim = len(vocab)
            self.embed_dim = hidden_dim
            self.feats_dim = additional_feat

        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        #Word Embeddings
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

    def load_params(self, checkpoint):
        self.word2idx = checkpoint['word2idx']
        self.ind2word = checkpoint['ind2word']

    def save_model(self, checkpt_file, params):
        params['word2idx'] = self.word2idx
        params['ind2word'] = self.ind2word
        checkpt_file = '{}_hidden{}_vocab{}_embed{}_feats{}_dropout{}.mdl'.format(checkpt_file, self.hidden_dim,
                                                                    self.vocab_dim, self.embed_dim,
                                                                    self.feats_dim, self.dropout_p)
        super(LanguageModel, self).save_model(checkpt_file, params)

    def forward(self, ref=None, word_idx=None, parameters=None):

        if ref is not None:
            sentence = ref['vocab_tensor'][:-1]
        elif word_idx is not None:
            sentence = torch.LongTensor([word_idx], device=self.device, requires_grad=True)
        else:
            raise ValueError('LanguageModel.forward must have either a ref or word_idx input')

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

    def targets(self, ref):
        return ref['vocab_tensor'][1:]

    def clear_gradients(self):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden()

    def train(self, n_epochs, instances, checkpt_file):
        return super(LanguageModel, self).train(n_epochs, instances, checkpt_file)

    def generate(self, feats):
        sentence = []
        word_idx = self.word2idx['<bos>']
        end_idx = self.word2idx['<eos>']

        with torch.no_grad():
            self.init_hidden()

            while word_idx != end_idx and len(sentence) < 100:
                output = self(word_idx=word_idx)
                word_idx = torch.argmax(output)
                sentence.append(self.ind2word[word_idx])

        return sentence

    def sent2vocab(self, refer):
        begin_index = self.word2idx['<bos>']
        end_index = self.word2idx['<eos>']
        unk_index = self.word2idx['<unk>']

        for sentence in refer.Sents.values():
            sentence['vocab'] = [begin_index]
            for token in sentence['tokens']:
                if token in self.word2idx:
                    sentence['vocab'].append(self.word2idx[token])
                else:
                    sentence['vocab'].append(unk_index)
            sentence['vocab'].append(end_index)

            sentence['vocab_tensor'] = torch.LongTensor(sentence['vocab'], device=self.device)


# Helper functions
def find_vocab(refer, threshold=0):
    vocab = {}

    for sentence in refer.Sents.values():
        for token in sentence['tokens']:
            if token in vocab:
                vocab[token] = vocab[token]+1
            else:
                vocab[token] = 1

    vocab = {token:value for (token,value) in vocab.items() if value > threshold}
    return list(vocab.keys())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test')
    parser.add_argument('checkpoint_file',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

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

    refer = REFER(args.data_root, args.dataset, args.splitBy)
    vocab = find_vocab(refer)
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    if (os.path.isfile(args.checkpoint_file)):
        model = LanguageModel(checkpt_file=args.checkpoint_file, use_cuda=use_cuda)
    else:
        model = LanguageModel(vocab=vocab, hidden_dim=args.hidden_dim,
                              use_cuda=use_cuda, dropout=args.dropout)

    # Preprocess REFER dataset
    model.sent2vocab(refer)

    if(args.mode == 'train'):
        print("Start Training")
        total_loss = model.train(args.epochs, refer.loadSents(refer.getRefIds(split='train')), args.checkpoint_file)

    if(args.mode == 'test'):
        print("Start Testing")
        print(model.generate([]))
