import json, math, random, argparse

import torch
torch.manual_seed(1)

### Data Preprocessing
class Stratifier:
    def __init__(self, answers_file, label_file, percent_test=0.8):
        print("Loading label dictionary")
        self.labels, self.label_to_idx = self.load_labels(label_file)

        print("Loading examples")
        self.image_obj_list = self.load_examples(answers_file)

        print("Stratified Split")
        self.train, self.test = self.split(percent_test)

    def load_examples(self, answers_file):
        with open(answers_file, 'r') as f:
            answers = json.load(f)

        n_answers = 0
        for answer_i in range(len(answers)):
            if type(answers[answer_i]) == dict:
                n_answers += 1
            else:
                n_answers += len(answers[answer_i])

        image_obj_list = {}
        objs = []

        for answer_i in range(len(answers)):
            answer_group = answers[answer_i]
            if len(answer_group) == 0:
                continue
            if type(answer_group) == dict:
                answer_group = [answer_group]

            all_labels = list(set([a for answer in answer_group for a in answer['multiple_choice_answer']]))
            all_labels.sort()
            image_obj_list[answer_group[0]['image_id']] = all_labels
            objs.extend(all_labels)

        return image_obj_list

    def count_freq(self, image_ids):
        freq_obj = torch.LongTensor(len(self.label_to_idx), 1).zero_()
        for j in image_ids:
            for i in self.image_obj_list[j]:
                freq_obj[self.label_to_idx[i]] += 1
        return freq_obj

    def split(self, percent_test):
        train = []
        test = []

        #Create binary tree branching on most frequent object
        freq_obj = self.count_freq(self.image_obj_list.keys())
        sorted, indices = torch.sort(freq_obj, 0, descending=True)
        indices = [i[0] for i in indices.tolist()]
        strata = {self.labels[indices[0]]: [v for v in self.image_obj_list.keys() if self.labels[indices[0]] in self.image_obj_list[v]],
                  'NOT': [v for v in self.image_obj_list.keys() if not self.labels[indices[0]] in self.image_obj_list[v]]}
        strata = self.rec_split(strata, [indices[0]])

        #Flatten tree
        strata_list = self.rec_flatten(strata)

        for s in strata_list:
            random.shuffle(s[1])
            split_idx = math.ceil(len(s[1])*percent_test)
            train.extend(s[1][0:split_idx])
            test.extend(s[1][split_idx:])

        return train, test

    def rec_split(self, strata, used):
        for key in strata.keys():
            if len(strata[key]) > 5:
                #Find Split Obj
                split_values = strata[key]
                freq_obj = self.count_freq(split_values)

                sorted, indices = torch.sort(freq_obj, 0, descending=True)
                indices = [i[0] for i in indices.tolist() if not i[0] in used]
                used.append(indices[0])

                #Don't keep going if all the objects are infrequent
                if (freq_obj[indices[0]]<=5).all():
                    continue

                strata[key] = {self.labels[indices[0]]: [v for v in split_values if self.labels[indices[0]] in self.image_obj_list[v]],
                               'NOT': [v for v in split_values if not self.labels[indices[0]] in self.image_obj_list[v]]}
                strata[key] = self.rec_split(strata[key], used)

        return strata

    def rec_flatten(self, tree):
        tree_list = []
        for key in tree.keys():
            if type(tree[key]) == list:
                tree_list.append([key, tree[key]])
            else:
                subtree_list = self.rec_flatten(tree[key])
                for l in subtree_list:
                    l[0] = "{}_{}".format(key,l[0])
                tree_list.extend(subtree_list)

        return tree_list

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = f.read().split('\n')

        label_to_idx = dict(zip(labels, range(0, len(labels))))

        return labels, label_to_idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split the data ')
    parser.add_argument('answers_file', help='JSON file containing the answers')
    parser.add_argument('label_file', help='text file with one class label per line')
    parser.add_argument('save_prefix', help='file path and prefix for output files')
    args = parser.parse_args()

    data = Stratifier(args.answers_file, args.label_file)

    with open("{}_train.txt".format(args.save_prefix), 'w') as f:
        f.write("\n".join([str(n) for n in data.train]))

    with open("{}_test.txt".format(args.save_prefix), 'w') as f:
        f.write("\n".join([str(n) for n in data.test]))