import os.path, argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

torch.manual_seed(1)

from ReferExpressionDataset import ReferExpressionDataset

#Network Definition
class LanguageModel(nn.Module):

    def __init__(self, vocab=None, hidden_dim=None, dropout=0,
            use_cuda=False, additional_feat=0):
        super(LanguageModel, self).__init__()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.embed_dim = hidden_dim
        self.feats_dim = additional_feat

        #Word Embeddings
        self.word2idx = dict(zip(vocab, range(len(vocab))))
        self.ind2word = vocab
        self.vocab_dim = len(vocab)
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.embed_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout1 = nn.Dropout(p=self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + self.feats_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(p=self.dropout_p)
        self.hidden2vocab = nn.Linear(self.hidden_dim, self.vocab_dim)
        self.hidden = self.init_hidden(1)

        self.to(self.device)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True))

    def run_debug(self,refer_dataset, batch_size=4):
        refer_dataset.active_split = 'train'

        if self.use_cuda:
            dataloader = DataLoader(refer_dataset, batch_size, shuffle=True)
        else:
            dataloader = DataLoader(refer_dataset, batch_size, shuffle=True, num_workers=4)


        self.train()
        refer_dataset.active_split = 'train'

        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch == 2388:
                print(sample_batched['vocab_tensor'])
                instances, targets = self.trim_batch(sample_batched)

                self.clear_gradients(batch_size)
                label_scores = self(instances)
                return

    def forward(self, ref=None, parameters=None):
        sentence = ref['vocab_tensor'][:, :-1]
        embeds = self.embedding(sentence)
        embeds = self.dropout1(embeds).permute(1, 0, 2)
        n, m, b = embeds.size()

        if 'feats' in ref:
            feats = ref['feats'].repeat(n, 1, 1)

            #Concatenate text embedding and additional features
            embeds = torch.cat([embeds, feats], 2)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout2(lstm_out)
        vocab_space = self.hidden2vocab(lstm_out)
        return vocab_space

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def trim_batch(self, ref):
        ref['vocab_tensor'] = ref['vocab_tensor'][:, torch.sum(ref['vocab_tensor'], 0) > 0]
        target = torch.tensor(ref['vocab_tensor'][:, 1:], dtype=torch.long, requires_grad=False, device=self.device)
        return ref, target

    def clear_gradients(self, batch_size):
        self.zero_grad()
        self.hidden = self.init_hidden(batch_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/')
    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
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

    model = LanguageModel(vocab=vocab, hidden_dim=args.hidden_dim, use_cuda=use_cuda, dropout=args.dropout)

    total_loss = model.run_debug(refer)

