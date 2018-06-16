from torch import FloatTensor, Size
from torch.nn import Embedding

class WordEmbedding():
    def __init__(self, text_file=None, vocab=None, dim=None, use_cuda=False):
        if text_file is not None:
            with open(text_file, 'r') as f:
                self.n_vocab, self.dim = [int(s) for s in f.readline().split()]
                self.embeddings = Embedding(self.n_vocab, self.dim)
                self.index = {}
                for i in range(self.n_vocab):
                    line_delim = f.readline().split()
                    embed = FloatTensor([float(s) for s in line_delim[1:]])

                    if(embed.size() == Size([self.dim])):
                        self.embeddings.weight[i, :] = embed
                        self.index[line_delim[0]] = i

        else:
            self.n_vocab = len(vocab)
            self.dim = dim
            self.embeddings = Embedding(self.n_vocab, self.dim)
            self.word2idx = dict(zip(vocab, range(len(vocab))))
            self.ind2word = vocab

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.embeddings.cuda()

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

