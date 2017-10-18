import torch
import torch.nn as nn

word2vec_file = '/Users/Mauceri/Datasets/books_in_sentences/books_p1_unk_vectors.txt'
save_file = '/Users/Mauceri/Datasets/books_in_sentences/word2vec_pytorch.pkl'

word_to_idx = {}
weights = []
with open(word2vec_file, 'r', encoding='utf-8', errors='replace') as f:
    vocab_size, feature_dim = [int(token) for token in f.readline().split()]

    line_num = 0
    for line in f:
        try:
            line_tokens = line.split()
            num_tokens = [float(token) for token in line_tokens[1:]]
            if(len(num_tokens) < feature_dim):
                continue
            word_to_idx[line_tokens[0]] = line_num
            line_num += 1
            weights.append(num_tokens)
        except:
            continue

if not '<UNK>' in word_to_idx:
    word_to_idx['<UNK>'] = line_num
    weights.append([0.0]*feature_dim)
    vocab_size += 1

embeds = nn.Embedding(vocab_size, feature_dim)
embeds.weight = nn.Parameter(torch.Tensor(weights))

word2vec = {}
word2vec['embed'] = embed
word2vec['index'] = word_to_idx

with open(save_file, 'wb') as save_f:
    pickle.dump(word2vec, save_f)
