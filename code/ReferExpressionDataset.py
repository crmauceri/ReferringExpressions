import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageStat import Stat

from refer_python3.refer import REFER

class ReferExpressionDataset(Dataset):

    def __init__(self, imagedir, dataroot, dataset, splitBy, vocab, use_cuda=False, transform_size=224, image_mean=[0.485, 0.456, 0.406],
                             image_std=[0.229, 0.224, 0.225], use_image=False, n_contrast_object=0):

        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.active_split = None
        self.use_image = use_image
        self.n_contrast_object = n_contrast_object

        self.refer = REFER(dataroot, dataset, splitBy)
        self.max_sent_len = max([len(sent['tokens']) for sent in self.refer.Sents.values()]) + 2 #For the begining and end tokens
        self.word2idx = dict(zip(vocab, range(1, len(vocab)+1)))
        self.sent2vocab(self.word2idx)

        self.root_dir = imagedir
        self.image_size = transform_size
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean,
                                 std=image_std)
            ])

        self.index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']]
        self.train_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'train']
        self.val_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'val']
        self.test_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'test']

    def __len__(self):
        return self.length(self.active_split)

    def __getitem__(self, item):
        return self.getItem(item, self.active_split)

    def length(self, split=None):
        if split is None:
            return len(self.index)
        elif split == 'train':
            return len(self.train_index)
        elif split == 'test':
            return len(self.test_index)
        elif split == 'val':
            return len(self.val_index)

    def getItem(self, idx, split=None, display_image=False):
        sample = {}

        if split is None:
            sent_idx = self.index[idx]
        elif split == 'train':
            sent_idx = self.train_index[idx]
        elif split == 'test':
            sent_idx = self.test_index[idx]
        elif split == 'val':
            sent_idx = self.val_index[idx]

        sentence = self.refer.Sents[sent_idx]['vocab_tensor']
        sample['tokens'] = self.refer.Sents[sent_idx]['tokens']
        sample['vocab_tensor'] = sentence

        if self.use_image or display_image:
            ref = self.refer.sentToRef[sent_idx]

            img_name = os.path.join(self.root_dir,
                                    self.refer.Imgs[ref['image_id']]['file_name'])
            image = Image.open(img_name)
            w, h = image.size

            if image.mode != "RGB":
                image = image.convert("RGB")

            bbox = self.refer.Anns[ref['ann_id']]['bbox']
            sample['object'], sample['pos'] = self.getObject(image, bbox)

            if self.n_contrast_object > 0:
                annIds = self.refer.getAnnIds(image_ids=ref['image_id'])
                bboxes = [self.refer.Anns[id]['bbox'] for id in annIds if id != ref['ann_id']]
                bboxes = random.sample(bboxes, min(self.n_contrast_object, len(bboxes)))
                sample['contrast'] = []
                for box in bboxes:
                    object, pos = self.getObject(image, box)
                    sample['contrast'].append({'object':object, 'pos':pos})

            image = self.standarizeImageFormat(image)
            sample['image'] = self.img_normalize(image)
            if display_image:
                sample['PIL'] = image

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

            padding = [0.0]*(self.max_sent_len - len(sentence['vocab']))
            sentence['vocab_tensor'] = torch.tensor(padding + sentence['vocab'], dtype=torch.long, device=self.device)

    def standarizeImageFormat(self, image, bbox=None):

        #Crop if bbox is smaller than image size
        if bbox is not None:
            width = int(bbox[2])
            height = int(bbox[3])
            bottom = int(bbox[1])
            left = int(bbox[0])

            image = image.crop((left, bottom, left+width, bottom+height))

        #Scale to self.image_size
        width, height = image.size
        ratio = float(self.image_size) / max([width, height])
        new_size = tuple([int(x*ratio) for x in [width, height]])
        image = image.resize(new_size, Image.ANTIALIAS)

        #Pad with mean value
        stat = Stat(image)
        pad_image = Image.new('RGB',  (self.image_size, self.image_size), tuple([int(x) for x in stat.mean]))
        pad_image.paste(image, (int((self.image_size-new_size[0])/2), int((self.image_size-new_size[1])/2)))

        return pad_image

    def getObject(self, image, bbox):
        w, h = image.size

        pos = torch.tensor(
            [bbox[0] / w, (bbox[1] + bbox[3]) / h, (bbox[0] + bbox[2]) / w, bbox[1] / h, (bbox[2] * bbox[3]) / (w * h)],
            dtype=torch.float, device=self.device)
        object = self.img_normalize(self.standarizeImageFormat(image, bbox))

        return object, pos
