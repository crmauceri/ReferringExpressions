import json, pickle, argparse, re

from anytree import Node
from anytree.iterators import PreOrderIter

import torch
import torch.autograd as autograd
import torch.nn as nn

torch.manual_seed(1)


class Instance:
    def __init__(self, query, answer, pos, tree_idx, image_idx):
        self.query = query
        self.answer = answer
        self.pos = pos
        self.tree_idx = tree_idx
        self.image_id = image_idx

    def vectorizeFeatures(self, word_to_idx, label_to_idx, tags_to_idx):
        if len(self.query) > 0 and len(self.answer) == 1:

            lost_words = [s for s in self.query if not s.lower() in word_to_idx and s != '<QUERY>']
            if len(lost_words) > 0:
                print(lost_words)

            self.flags = autograd.Variable(torch.FloatTensor([[1] if w == '<QUERY>' else [0] for w in self.query]))
            self.inputs = autograd.Variable(torch.LongTensor(
                [word_to_idx[w.lower()] if w.lower() in word_to_idx else word_to_idx['<UNK>'] for w in self.query]))
            self.targets = autograd.Variable(torch.LongTensor([label_to_idx[l] for l in self.answer]))
            pos_tensor = torch.FloatTensor(len(self.pos), len(tags_to_idx)).zero_()
            for pos_i in range(len(self.pos)):
                pos_tensor[pos_i][tags_to_idx[self.pos[pos_i]]] = 1
            self.pos_variable = autograd.Variable(pos_tensor)


# Data Preprocessing
class DataLoader:
    def __init__(self, queries_file, answers_file, word2vec_file, tag_file, label_file, image_idx_file, tree_file):
        print("Loading labels")
        self.labels, self.label_to_idx = self.load_labels(label_file)

        print("Loading examples")
        self.instances = self.load_examples(queries_file, answers_file, image_idx_file)

        if not tree_file is None:
            print("Parse trees")
            self.add_trees(tree_file, self.instances)

        print("Loading word embeddings")
        self.word_to_idx, self.embed = self.load_word2vec(word2vec_file)

        print("Loading tag dictionary")
        with open(tag_file, 'r') as f:
            tags = f.read().split('\n')
            tags = list(set([t[1:].strip() if t[0] == '-' else t.strip() for t in tags]))
            tags.sort()
            tags.extend(['.', ',', 'XX', ' ', ':', "''", "$", "-LRB-", "-RRB-"])
        self.tags_to_idx = dict(zip(tags, range(len(tags))))
        self.tags = tags

        print("Preprocessing examples")
        self.prepare_data(self.instances, self.word_to_idx, self.label_to_idx, self.tags_to_idx)

    @staticmethod
    def load_examples(queries_file, answers_file, image_idx_file=None):
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        with open(answers_file, 'r') as f:
            answers = json.load(f)

        if image_idx_file is None:
            use_image_idxs = False
        else:
            use_image_idxs = True
            with open(image_idx_file, 'r') as f:
                image_idxs = [int(n) for n in f.read().split()]

        instances = []

        for i in range(len(queries)):
            flat_pos = []
            flat_tree = []

            query = queries[i][0] if type(queries[i]) == list and len(queries[i]) > 0 else queries[i]
            if len(query) == 0:
                continue
            if use_image_idxs and not query['image_id'] in image_idxs:
                continue

            if type(answers[i]) == list:
                flat_answer = [x['multiple_choice_answer'] for x in answers[i]]
            else:
                flat_answer = [answers[i]['multiple_choice_answer']]

            if type(queries[i]) == list:
                flat_question = [x['tokenized'] for x in queries[i]]
                if 'pos' in queries[i][0]:
                    flat_pos = [x['pos'] for x in queries[i]]
                if 'tree_idx' in queries[i][0]:
                    flat_tree = [x['tree_idx'] for x in queries[i]]
            else:
                flat_question = [queries[i]['tokenized']]
                if 'pos' in queries[i]:
                    flat_pos = [queries[i]['pos']]
                if 'tree_idx' in queries[i]:
                    flat_tree = [queries[i]['tree_idx']]

            for j in range(len(flat_question)):
                pos = []
                tree = []
                if len(flat_pos) > j:
                    pos = flat_pos[j]
                if len(flat_tree) > j:
                    tree = flat_tree[j]

                instances.append(
                    Instance(flat_question[j], flat_answer[j], pos, tree, query['image_id']))

        return instances

    @staticmethod
    def flatten(instances, idxs, key):
        flat_list = []
        for i in idxs:
            if type(instances[i])==list:
                flat_list.extend([x[key] for x in instances[i]])
            else:
                flat_list.append(instances[i][key])

        return flat_list

    @staticmethod
    def add_trees(tree_file, instances):
        trees = []
        with ParseDependancy(tree_file) as factory:
            nextTree = factory.nextTree(None)
            while nextTree != None:
                trees.append(nextTree)
                nextTree = factory.nextTree(None)

        for i in instances:
            inst_tree = trees[i.tree_idx-1]
            pos_tokens = [node.name.split('/')[-1] for node in
                          PreOrderIter(inst_tree, filter_=lambda n: n.is_leaf and n.token[0] != '*')]
            i.pos = pos_tokens

    @staticmethod
    def prepare_data(instances, word_to_idx, label_to_idx, tags_to_idx):
        for i in instances:
            i.vectorizeFeatures(word_to_idx, label_to_idx, tags_to_idx)

    @staticmethod
    def load_labels(label_file):
        with open(label_file, 'r') as f:
            labels = f.read().split('\n')

        label_to_idx = dict(zip(labels, range(0, len(labels))))

        return labels, label_to_idx

    @staticmethod
    def load_word2vec(word2vec_file):
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
        return word_to_idx, embeds


class ParseDependancy:
    def __init__(self, tree_file):
        self.tree_file = tree_file
        self.space_pattern = re.compile('\s')

    def __enter__(self):
        self.fp = open(self.tree_file, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()

    def nextTree(self, parent):
        tree = Node('', parent, token=None)
        pOpen = True

        if parent == None:
            nextChar = ''
            while nextChar != '(':
                nextChar = self.fp.read(1)
                if nextChar == '':
                    # End of file
                    return None

        value = ''
        while pOpen:
            nextChar = self.fp.read(1)
            if nextChar == '(':
                self.nextTree(tree)
            elif nextChar == ')':
                if len(value) > 0:
                    tree.token = value
                pOpen = False
            elif nextChar == '':
                # Ran out of characters
                return None
            elif re.match(self.space_pattern, nextChar):
                tree.name = value
                value = ''
            else:
                value += nextChar
        return tree

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    indices = []
    for i, d in enumerate(data.instances):
        if not hasattr(d, 'inputs') or not hasattr(d, 'targets'):
            continue
        else:
            indices.append(i)
    data.indices = indices

    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess the data for SimpleClassifier.')
    parser.add_argument('queries_file', help='JSON file containing the queries')
    parser.add_argument('answers_file', help='JSON file containing the answers')
    parser.add_argument('label_file', help='text file with one class label per line')
    parser.add_argument('tag_file', help='text file with one part of speech tag per line')
    parser.add_argument('word2vec_file', help='text file with one vocab word and one trained word2vec vector per line')
    parser.add_argument('save_file', help='output is saved as pickle file at this location')
    parser.add_argument('--image_idx_file', dest='image_idx_file', type=str, default=None,
                        help='Text file with one image_idx per line, selects images to include in preprocessed data. Default: use all images')
    parser.add_argument('--tree_file', dest='tree_file', type=str, default=None,
                        help='Text file with parse trees. Default: use all images')
    args = parser.parse_args()

    data = DataLoader(args.queries_file, args.answers_file, args.word2vec_file, args.tag_file, args.label_file,
                      args.image_idx_file, tree_file=args.tree_file)

    with open(args.save_file, 'wb') as save_f:
        pickle.dump(data, save_f)