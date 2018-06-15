import os.path, argparse, re

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

from ClassifierHelper import Classifier
from WordEmbedding import WordEmbedding, find_vocab

from pyutils.refer_python3.refer import REFER

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, checkpt_file=None, word_embedding=None, hidden_dim=None, use_cuda=False):
        super(LanguageModel, self).__init__(use_cuda)

        if checkpt_file is not None:
            m = re.match('hidden(?P<hidden>\d+)_vocab(?P<vocab>\d+)_embed(?P<embed>\d+)', checkpt_file)
            self.hidden_dim = m.group('hidden')
            vocab_dim = m.group('vocab')
            embed_dim = m.group('embed')
        else:
            self.hidden_dim = hidden_dim
            self.word_embeddings = word_embedding
            vocab_dim = self.word_embeddings.n_vocab
            embed_dim = self.word_embeddings.dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim)
        self.hidden2vocab = nn.Linear(self.hidden_dim, vocab_dim)
        self.hidden = self.init_hidden()

        if self.use_cuda:
            self.cuda()

        if checkpt_file is not None:
            super(LanguageModel, self).load_model(checkpt_file)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def load_params(self, checkpoint):
        self.word_embeddings = checkpoint['word_embeddings']

    def save_model(self, checkpt_file, params):
        params['word_embeddings'] = self.word_embeddings
        checkpt_file = '{}_hidden{}_vocab{}_embed{}.mdl'.format(checkpt_file, self.hidden_dim,
                                                                self.word_embeddings.n_vocab, self.word_embeddings.dim)
        super(LanguageModel, self).save_model(checkpt_file, params)

    def forward(self, ref, parameters):
        sentence = torch.LongTensor(ref['vocab'][:-1])

        if self.use_cuda:
            sentence = sentence.cuda()

        embeds = self.word_embeddings.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        vocab_space = self.hidden2vocab(lstm_out.view(len(sentence), -1))
        vocab_scores = F.log_softmax(vocab_space, dim=1)
        return vocab_scores

    def targets(self, ref):
        return torch.LongTensor(ref['vocab'][1:])

    def clear_gradients(self):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden()

    def train(self, n_epochs, instances, checkpt_file):
        return super(LanguageModel, self).train(n_epochs, instances, checkpt_file)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test')
    parser.add_argument('checkpoint_file',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')

    parser.add_argument('--word_embedding_file', dest='word_embedding_file',
                        help='Location of trained word embeddings')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=100,
                        help='Size of LSTM embedding (Default:100)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    refer = REFER(args.data_root, args.dataset, args.splitBy)

    if args.word_embedding_file is not None:
        word_embedding = WordEmbedding(word_embedding=args.word_embedding_file)
    else:
        vocab = find_vocab(refer)
        #Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])
        word_embedding = WordEmbedding(index=dict(zip(vocab, range(len(vocab)))), dim=200)

    word_embedding.sent2vocab(refer)

    if (os.path.isfile(args.checkpoint_file)):
        model = LanguageModel(checkpt_file=args.checkpoint_file, use_cuda=use_cuda)
    else:
        model = LanguageModel(word_embedding=word_embedding, hidden_dim=args.hidden_dim, use_cuda=use_cuda)

    if(args.mode == 'train'):
        print("Start Training")
        total_loss = model.train(args.epochs, refer.loadSents(refer.getRefIds(split='train')), args.checkpoint_file)

    if(args.mode == 'test'):
        print("Start Testing")
        model.test(refer.loadRefs(refer.getRefIds(split='test')))