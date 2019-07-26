import torch
import torch.nn as nn

#torch.manual_seed(1)

from .ClassifierHelper import Classifier, SequenceLoss

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, cfg): #checkpt_file=None, vocab=None, hidden_dim=None, dropout=0, additional_feat=0):
        super(LanguageModel, self).__init__(cfg, loss_function = SequenceLoss(nn.CrossEntropyLoss()))

        self.feats_dim = cfg.IMG_NET.FEATS
        self.hidden_dim = cfg.LSTM.HIDDEN
        self.embed_dim = cfg.LSTM.EMBED
        self.dropout_p = cfg.TRAINING.DROPOUT
        self.l2_fraction = cfg.TRAINING.L2_FRACTION

        #Word Embeddings
        with open(cfg.DATASET.VOCAB, 'r') as f:
            vocab = f.read().split()
        # Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])

        self.word2idx = dict(zip(vocab, range(1, len(vocab)+1)))
        self.ind2word = ['<>'] + vocab
        self.vocab_dim = len(vocab)+1
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.embed_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout1 = nn.Dropout(p=self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + self.feats_dim, self.hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(p=self.dropout_p)
        self.hidden2vocab = nn.Linear(self.hidden_dim, self.vocab_dim)
        self.hidden = self.init_hidden(1)

        self.to(self.device)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True))

    def forward(self, ref=None):
        sentence = ref['vocab_tensor'][:, :-1]
        embeds = self.embedding(sentence)
        embeds = self.dropout1(embeds)
        n, m, b = embeds.size()

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
        vocab_space = self.hidden2vocab(lstm_out)
        return vocab_space

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device).unsqueeze(0)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def trim_batch(self, ref):
        ref['vocab_tensor'] = ref['vocab_tensor'][:, torch.sum(ref['vocab_tensor'], 0) > 0]
        target = ref['vocab_tensor'][:, 1:].clone().detach()
        return ref, target

    def clear_gradients(self, batch_size):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden(batch_size)

    def generate(self, start_word='<bos>', feats=None, max_len=30):
        sentence = []
        word_idx = self.word2idx[start_word]
        end_idx = self.word2idx['<eos>']

        self.clear_gradients(batch_size=1)

        idx = 0
        with torch.no_grad():
            while word_idx != end_idx and len(sentence) < max_len:
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                word_idx = torch.argmax(output)

                if word_idx != end_idx:
                    sentence.append(self.ind2word[word_idx])
                idx += 1

        return sentence

    def generate_batch(self, start_word='<bos>', feats=None, max_len=30):
        tensor = torch.zeros((feats.shape[0], max_len, self.vocab_dim), device=self.device)
        word_idx = self.word2idx[start_word]

        self.clear_gradients(batch_size=feats.shape[0])

        with torch.no_grad():
            for idx in range(max_len):
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                tensor[:, idx, :] = output.squeeze(1)

        return tensor

    def test(self, instance):
        return self.generate(instance=instance)
