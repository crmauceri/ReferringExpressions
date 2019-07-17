import re

import torch
import torch.nn as nn

#torch.manual_seed(1)

from .ClassifierHelper import Classifier, SequenceLoss

#Network Definition
class LanguageModel(Classifier):

    def __init__(self, cfg):
        super(LanguageModel, self).__init__(cfg, loss_function = SequenceLoss(nn.CrossEntropyLoss()))

        self.feats_dim = cfg.IMG_NET.FEATS
        self.hidden_dim = cfg.LSTM.HIDDEN
        self.embed_dim = cfg.LSTM.EMBED
        self.dropout_p = cfg.TRAINING.DROPOUT
        self.l2_fraction = cfg.TRAINING.L2_FRACTION

        # Word Embeddings
        with open(cfg.DATASET.VOCAB, 'r') as f:
            vocab = f.read().split()
        # Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])

        self.word2idx = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.ind2word = ['<>'] + vocab
        self.vocab_dim = len(vocab) + 1
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.embed_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout1 = nn.Dropout(p=self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + self.feats_dim, self.hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(p=self.dropout_p)
        self.hidden2vocab = nn.Linear(self.hidden_dim, self.vocab_dim)
        self.hidden = self.init_hidden(1)

        self.txt_layernorm1 = nn.LayerNorm(self.hidden_dim)
        self.txt_layernorm2 = nn.LayerNorm(self.hidden_dim)

        self.to(self.device)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device, requires_grad=True))

    @staticmethod
    def get_checkpt_file(checkpt_file, hidden_dim, feats_dim, dropout_p):
        return '{}_hidden{}_feats{}_dropout{}.mdl'.format(checkpt_file, hidden_dim, feats_dim, dropout_p)

    def checkpt_file(self, checkpt_file):
        return self.get_checkpt_file(checkpt_file, self.hidden_dim, self.feats_dim, self.dropout_p)

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

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device).unsqueeze(0)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def trim_batch(self, ref):
        ref['vocab_tensor'] = ref['vocab_tensor'][:, torch.sum(ref['vocab_tensor'], 0) > 0]
        target = torch.tensor(ref['vocab_tensor'][:, 1:], dtype=torch.long, requires_grad=False, device=self.device)
        return ref, target

    def clear_gradients(self, batch_size):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden(batch_size)

    def generate(self, start_word='<bos>', instance=None, feats=None):
        sentence = []
        word_idx = self.word2idx[start_word]
        end_idx = self.word2idx['<eos>']

        with torch.no_grad():
            self.hidden = self.init_hidden(1)

            while word_idx != end_idx and len(sentence) < 30:
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                word_idx = torch.argmax(output)
                sentence.append(self.ind2word[word_idx])

        return sentence

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
#     parser.add_argument('mode', help='train/test')
#     parser.add_argument('checkpoint_prefix',
#                         help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')
#
#     parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/')
#     parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
#     parser.add_argument('--dataset', help='dataset name', default='refcocog')
#     parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
#     parser.add_argument('--epochs', dest='epochs', type=int, default=1,
#                         help='Number of epochs to train (Default: 1)')
#     parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
#                         help='Size of LSTM embedding (Default:100)')
#     parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')
#
#     args = parser.parse_args()
#
#     use_cuda = torch.cuda.is_available()
#
#     with open('vocab_file.txt', 'r') as f:
#         vocab = f.read().split()
#     # Add the start and end tokens
#     vocab.extend(['<bos>', '<eos>', '<unk>'])
#
#     refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_cuda)
#
#     checkpt_file = LanguageModel.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 0, args.dropout)
#     if (os.path.isfile(checkpt_file)):
#         model = LanguageModel(checkpt_file=checkpt_file, vocab=vocab, use_cuda=use_cuda)
#     else:
#         model = LanguageModel(vocab=vocab, hidden_dim=args.hidden_dim,
#                                   use_cuda=use_cuda, dropout=args.dropout)
#
#     if(args.mode == 'train'):
#         print("Start Training")
#         total_loss = model.run_training(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image':False})
#
#     if(args.mode == 'test'):
#         print("Start Testing")
#         print(model.generate([]))
