import torch
import torchvision

import os.path as osp
import json
import pickle

from maskrcnn_benchmark.data.datasets.coco import COCODataset



class ReferExpressionDataset(COCODataset):
    def __init__(
        self, ann_file, ann_root, referdataset, vocabfile, remove_images_without_annotations, \
            transforms=None, disable_cuda=False,
    ):
        super(ReferExpressionDataset, self).__init__(ann_file, ann_root, remove_images_without_annotations, transforms)

        # Fix the image ids assigned by the torchvision dataset loader
        self.ids = dict(zip(self.coco.imgs.keys(), self.coco.imgs.keys()))

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.active_split = None

        with open(vocabfile, 'r') as f:
            self.vocab = [v.strip() for v in f.readlines()]


        self.vocab.extend(['<bos>', '<eos>', '<unk>'])
        self.word2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))

        self.createRefIndex(referdataset)

        # if dataset == 'refcocog':
        #     self.unique_test_objects = [ref['sent_ids'][0] for key, ref in self.refer.annToRef.items() if
        #                                 ref['split'] == 'val']
        # else:
        #     self.unique_test_objects = [ref['sent_ids'][0] for key, ref in self.refer.annToRef.items() if
        #                                 ref['split'] == 'test']

    def __len__(self):
        return self.length(self.active_split)

    def length(self, split=None):
        if split is None:
            return len(self.index)
        elif split == 'train':
            return len(self.train_index)
        elif split == 'test':
            return len(self.test_index)
        elif split == 'test_unique':
            return len(self.unique_test_objects)
        elif split == 'val':
            return len(self.val_index)

    def __getitem__(self, item):
        return self.getItem(item, self.active_split)

    def getItem(self, idx, split=None):

        if split is None:
            sent_idx = self.index[idx]
        elif split == 'train':
            sent_idx = self.train_index[idx]
        elif split == 'test':
            sent_idx = self.test_index[idx]
        elif split == 'test_unique':
            sent_idx = self.unique_test_objects[idx]
        elif split == 'val':
            sent_idx = self.val_index[idx]

        ref = self.coco.sentToRef[sent_idx]
        img, target = super(COCODataset, self).__getitem__(ref['image_id'])

        sent = self.coco.sents[sent_idx]
        if not 'vocab_tensor' in self.coco.sents[sent_idx]:
            padding = [0.0] * (self.max_sent_len - len(sent['vocab']))
            self.coco.sents[sent_idx]['vocab_tensor'] = torch.tensor(padding + sent['vocab'], dtype=torch.long,
                                                device=self.device)

        return img, target, sent

    def createRefIndex(self, ref_file):

        with open(ref_file, 'rb') as f:
            refs = pickle.load(f)

        print('creating index...')

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}

        refs = [ref for ref in refs if ref['ann_id'] in self.coco.anns]
        for ref in refs:
            # ids
            ref_id = ref['ref_id']

            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = self.coco.anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                self.sent2vocab(sent)
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.coco.refs = Refs
        self.coco.imgToRefs = imgToRefs
        self.coco.refToAnn = refToAnn
        self.coco.annToRef = annToRef
        self.coco.catToRef = catToRefs
        self.coco.sents = Sents
        self.coco.sentToRef = sentToRef
        self.coco.sentToTokens = sentToTokens

        self.max_sent_len = max(
            [len(sent['tokens']) for sent in self.coco.sents.values()]) + 2  # For the begining and end tokens

        self.index = [sent_id for ref in self.coco.refs for sent_id in self.coco.refs[ref]['sent_ids']]
        self.train_index = [sent_id for ref in self.coco.refs for sent_id in self.coco.refs[ref]['sent_ids'] if
                            self.coco.refs[ref]['split'] == 'train']
        self.val_index = [sent_id for ref in self.coco.refs for sent_id in self.coco.refs[ref]['sent_ids'] if
                          self.coco.refs[ref]['split'] == 'val']
        self.test_index = [sent_id for ref in self.coco.refs for sent_id in self.coco.refs[ref]['sent_ids'] if
                        self.coco.refs[ref]['split'] == 'test']

    def sent2vocab(self, sent):
        begin_index = self.word2idx['<bos>']
        end_index = self.word2idx['<eos>']
        unk_index = self.word2idx['<unk>']

        sent['vocab'] = [begin_index]
        for token in sent['tokens']:
            if token in self.word2idx:
                sent['vocab'].append(self.word2idx[token])
            else:
                sent['vocab'].append(unk_index)
        sent['vocab'].append(end_index)