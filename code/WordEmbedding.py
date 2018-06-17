from torch import LongTensor, Size, device
from torch.nn import Embedding

class WordEmbedding():
    def __init__(self, vocab=None, dim=None, use_cuda=False):
        self.n_vocab = len(vocab)
        self.dim = dim
        self.embeddings = Embedding(self.n_vocab, self.dim)
        self.word2idx = dict(zip(vocab, range(len(vocab))))
        self.ind2word = vocab

        self.use_cuda = use_cuda
        if use_cuda:
            self.device = device('cuda')
        else:
            self.device = device('cpu')
        self.embeddings.to(self.device)


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

            sentence['vocab_tensor'] = LongTensor(sentence['vocab'], device=self.device)

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

