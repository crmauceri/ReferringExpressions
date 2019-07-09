import os, random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageStat import Stat
import numpy as np

class ReferExpressionDataset(Dataset):

    def __init__(self, refer, dataset, vocab, disable_cuda=False, transform_size=224, image_mean=[0.485, 0.456, 0.406],
                 image_std=[0.229, 0.224, 0.225], use_image=False, depth_mean=19018.9, depth_std=18798.8, use_depth=False, n_contrast_object=0):

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.active_split = None
        self.use_image = use_image
        self.use_depth = use_depth
        self.n_contrast_object = n_contrast_object

        self.refer = refer
        self.max_sent_len = max([len(sent['tokens']) for sent in self.refer.Sents.values()]) + 2 #For the begining and end tokens
        self.word2idx = dict(zip(vocab, range(1, len(vocab)+1)))
        self.sent2vocab(self.word2idx)

        self.toTensorTransform = transforms.ToTensor()

        self.image_size = transform_size
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean,
                                 std=image_std)
            ])

        self.depth_normalize = transforms.Compose([
            transforms.Normalize(mean=[depth_mean],
                                  std=[depth_std])
        ])


        # Use these to visualize tensors
        rev_mean = [-1.0*image_mean[0]/image_std[0], -1.0*image_mean[1]/image_std[1], -1.0*image_mean[2]/image_std[2]]
        rev_std = [1.0/image_std[0], 1.0/image_std[1], 1.0/image_std[2]]
        self.rev_img_normalize = transforms.Compose([
            transforms.Normalize(mean=rev_mean, std=rev_std),
            transforms.ToPILImage(mode='RGB')
        ])

        self.rev_depth_normalize = transforms.Compose([
            transforms.Normalize(mean=-1 * depth_mean / depth_std, std=1 / depth_std),
            transforms.ToPILImage(mode='I')
        ])

        self.index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']]
        self.train_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'train']
        self.val_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'val']
        self.test_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'test']

        if dataset == 'refcocog':
            self.unique_test_objects = [ref['sent_ids'][0] for key, ref in self.refer.annToRef.items() if
                                        ref['split'] == 'val']
        else:
            self.unique_test_objects = [ref['sent_ids'][0] for key,ref in self.refer.annToRef.items() if ref['split'] == 'test']

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
        elif split == 'test_unique':
            return len(self.unique_test_objects)
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
        elif split == 'test_unique':
            sent_idx = self.unique_test_objects[idx]
        elif split == 'val':
            sent_idx = self.val_index[idx]

        sentence = self.refer.Sents[sent_idx]['vocab_tensor']
        sample['tokens'] = self.refer.Sents[sent_idx]['tokens']
        sample['vocab_tensor'] = sentence

        sample['refID'] = self.refer.sentToRef[sent_idx]['ref_id']
        sample['imageID'] = self.refer.sentToRef[sent_idx]['image_id']
        sample['objectID'] = self.refer.sentToRef[sent_idx]['ann_id']

        if('zero_shot' in self.refer.sentToRef[sent_idx]):
            sample['zero-shot'] = self.refer.sentToRef[sent_idx]['zero_shot']
        else:
            sample['zero-shot'] = False

        if(self.refer.annToRef[sample['objectID']]['category_id'] in self.refer.Cats):
            sample['objectClass'] = self.refer.annToRef[sample['objectID']]['category_id']
        else:
            sample['objectClass'] = "unknown"

        if self.use_image or display_image:
            sample.update(self.getAllImageFeatures(sent_idx, display_image=display_image))

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
        image = image.copy()

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
        if image.mode == 'RGB':
            stat = Stat(image)
            median_val = tuple([int(x) for x in stat.median])
        elif image.mode == 'I':
            median_val = int(np.round(np.median(image, axis=0))[0])
        else:
            raise ValueError('Mode not supported')
        pad_image = Image.new(image.mode,  (self.image_size, self.image_size), median_val)
        pad_image.paste(image, (int((self.image_size-new_size[0])/2), int((self.image_size-new_size[1])/2)))

        return pad_image

    def normalizeDepth(self, raw_depth, bbox=None):
        depth = self.toTensorTransform(self.standarizeImageFormat(raw_depth, bbox=bbox))
        depth = self.depth_normalize(torch.as_tensor(depth, dtype=torch.float))
        return depth

    def getObject(self, bbox, image, depth=None):
        w, h = image.size

        pos = torch.tensor(
            [bbox[0] / w, (bbox[1] + bbox[3]) / h, (bbox[0] + bbox[2]) / w, bbox[1] / h, (bbox[2] * bbox[3]) / (w * h)],
            dtype=torch.float, device=self.device)
        rgb_object = self.img_normalize(self.standarizeImageFormat(image, bbox=bbox))

        if depth is not None:
            d_object = self.normalizeDepth(depth, bbox=bbox)
            rgbd_object = torch.cat((rgb_object, d_object), 0)
            return rgbd_object, pos
        else:
            return rgb_object, pos

    def getAllImageFeatures(self, sent_idx, display_image=False):
        sample = dict()

        ref = self.refer.sentToRef[sent_idx]

        # Load the image
        img_name = os.path.join(self.refer.IMAGE_DIR,
                                self.refer.Imgs[ref['image_id']]['file_name'])
        raw_image = Image.open(img_name)

        w, h = raw_image.size

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # Scale and normalize image
        image = self.img_normalize(self.standarizeImageFormat(raw_image))

        # Load depth image
        if self.use_depth:
            mode = 'RGBA'
            depth_name = os.path.join(self.refer.DEPTH_DIR,
                                      self.refer.Imgs[ref['image_id']]['depth_file_name'])
            raw_depth = Image.open(depth_name)

            # Scale and normalize depth image
            depth = self.normalizeDepth(raw_depth)
            sample['image'] = torch.cat((image, depth), 0)

        else:
            mode = 'RGB'
            raw_depth = None
            sample['image'] = image

        #Extract a crop of the target bounding box
        bbox = self.refer.Anns[ref['ann_id']]['bbox']
        sample['object'], sample['pos'] = self.getObject(image=raw_image, depth=raw_depth, bbox=bbox)

        # Extract crops of contrast objects
        if self.n_contrast_object > 0:
            annIds = self.refer.getAnnIds(image_ids=ref['image_id'])
            bboxes = [self.refer.Anns[id]['bbox'] for id in annIds if
                      id != ref['ann_id'] and int(self.refer.Anns[id]['bbox'][2]) > 0 and int(
                          self.refer.Anns[id]['bbox'][3]) > 0]
            bboxes = random.sample(bboxes, min(self.n_contrast_object, len(bboxes)))
            sample['contrast'] = []
            for bbox in bboxes:
                object, pos = self.getObject(image=raw_image, depth=raw_depth, bbox=bbox)
                sample['contrast'].append({'object': object, 'pos': pos})

        if display_image:
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw = ImageDraw.Draw(raw_image)
            draw.rectangle(bbox, fill=None, outline=(255, 0, 0, 255))
            del draw
            sample['PIL'] = raw_image

        return sample


# Test data loader
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from refer import REFER
    from torch.utils.data import DataLoader
    from PIL import Image
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Test dataset loading')
    parser.add_argument('--img_root', help='path to the image directory', default='datasets/SUNRGBD/images')
    parser.add_argument('--depth_root', help='path to the image directory', default='datasets/SUNRGBD/images')
    parser.add_argument('--data_root', help='path to data directory', default='datasets/sunspot/annotations/')
    parser.add_argument('--dataset', help='dataset name', default='sunspot')
    parser.add_argument('--version', help='team that made the dataset splits', default='boulder')
    args = parser.parse_args()

    with open('datasets/vocab_file.txt', 'r') as f:
        vocab = f.read().split()
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    refer = REFER(data_root=args.data_root, image_dir=args.img_root, depth_dir=args.depth_root, dataset=args.dataset, version=args.version)
    refer_dataset = ReferExpressionDataset(refer, args.dataset, vocab, use_image=True, use_depth=True, n_contrast_object=1)
    sample = refer_dataset.getItem(0, display_image=True)
    sample['PIL'].show()

    image = refer_dataset.rev_img_normalize(sample['image'][0:3, :, :])
    image.show()
    plt.imshow(sample['image'][3, :, :])
    plt.show()

    object_img = refer_dataset.rev_img_normalize(sample['object'][0:3, :, :])
    object_img.show()
    plt.imshow(sample['object'][3, :, :])
    plt.show()

    contrast_img = refer_dataset.rev_img_normalize(sample['contrast'][0]['object'][0:3, :, :])
    contrast_img.show()
    plt.imshow(sample['contrast'][0]['object'][3, :, :])
    plt.show()