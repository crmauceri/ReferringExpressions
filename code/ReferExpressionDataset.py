import os
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from refer_python3.refer import REFER

class ReferExpressionDataset(Dataset):

    def __init__(self, imagedir, dataroot, dataset, splitBy, vocab, use_cuda=False, transform=None):

        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.refer = REFER(dataroot, dataset, splitBy)
        self.word2idx = dict(zip(vocab, range(len(vocab))))
        self.sent2vocab(self.word2idx)

        self.root_dir = imagedir
        self.img_transform = transform
        self.obj_transform = transforms.Compose([CropBoundingBox, transform])

        self.index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']]
        self.train_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'train']
        self.val_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'val']
        self.test_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'test']


    def length(self, split=None):
        if split is None:
            return len(self.index)
        elif split == 'train':
            return len(self.train_index)
        elif split == 'test':
            return len(self.test_index)
        elif split == 'val':
            return len(self.val_index)

    def getItem(self, idx, split=None, use_image=False):

        if split is None:
            sent_idx = self.index[idx]
        elif split == 'train':
            sent_idx = self.train_index[idx]
        elif split == 'test':
            sent_idx = self.test_index[idx]
        elif split == 'val':
            sent_idx = self.val_index[idx]

        sentence = self.refer.Sents[sent_idx]['vocab_tensor']

        ref = self.refer.sentToRef[sent_idx]
        bbox = torch.tensor(self.refer.Anns[ref['ann_id']]['bbox'], dtype=torch.float, device=self.device)

        if use_image:
            img_name = os.path.join(self.root_dir,
                                    self.refer.Imgs[ref['image_id']]['file_name'])
            image = io.imread(img_name)

            sample = {'image': image, 'bbox': bbox}
            sample['object'] = self.obj_transform(sample)

            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            sample = {}

        sample['vocab_tensor'] = sentence
        return sample

    def sent2vocab(self, word2idx):
        begin_index = word2idx['<bos>']
        end_index = word2idx['<eos>']
        unk_index = word2idx['<unk>']

        for sentence in self.refer.Sents.values():
            sentence['vocab'] = [begin_index]
            for token in sentence['tokens']:
                if token in word2idx:
                    sentence['vocab'].append(word2idx[token])
                else:
                    sentence['vocab'].append(unk_index)
            sentence['vocab'].append(end_index)

            sentence['vocab_tensor'] = torch.tensor(sentence['vocab'], dtype=torch.long, device=self.device)


class CropBoundingBox(object):
    """Crop the image in a sample."""

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        h, w = image.shape[:2]
        new_h, new_w = sample['bbox'][2:]

        top = sample['bbox'][1]
        left = sample['bbox'][0]

        image = image[top: top + new_h,
                      left: left + new_w]

        return image
