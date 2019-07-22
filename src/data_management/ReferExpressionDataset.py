import os, random, json, math
from PIL import ImageDraw, Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageStat import Stat
import numpy as np

# Helper class to load images
class ImageProcessing:
    def __init__(self, cfg, img_root=None, depth_root=None, data_root=None): #img_dir, depth_dir, data_dir,
                 #disable_cuda=False, transform_size=224,
                 #image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], use_image=False,
                 #depth_mean=19018.9, depth_std=18798.8, use_depth=False):

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.use_image = cfg.IMG_PROCESSING.USE_IMAGE
        self.use_depth = cfg.IMG_PROCESSING.USE_DEPTH

        self.IMAGE_DIR = cfg.DATASET.IMG_ROOT
        self.DEPTH_DIR = cfg.DATASET.DEPTH_ROOT
        self.DATA_DIR = cfg.DATASET.DATA_ROOT

        if img_root is not None:
            self.IMAGE_DIR = img_root
        if depth_root is not None:
            self.DEPTH_DIR = depth_root
        if data_root is not None:
            self.DATA_DIR = data_root

        # load filepaths from data_root/depth.json
        if self.DEPTH_DIR is not None:
            depth_file = os.path.join(self.DATA_DIR, 'depth.json')
            depth = json.load(open(depth_file, 'r'))
            self.depth_map = {int(key):value for key,value in depth.items()}

        self.toTensorTransform = transforms.ToTensor()

        image_mean = cfg.IMG_PROCESSING.IMG_MEAN
        image_std = cfg.IMG_PROCESSING.IMG_STD
        self.image_size = cfg.IMG_PROCESSING.TRANSFORM_SIZE
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean,
                                 std=image_std)
            ])

        depth_mean = cfg.IMG_PROCESSING.DEPTH_MEAN
        depth_std = cfg.IMG_PROCESSING.DEPTH_STD
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

    def getAllImageFeatures(self, img_idx, file_name):
        sample = dict()

        # Load the image
        img_name = os.path.join(self.IMAGE_DIR, file_name)
        raw_image = Image.open(img_name)

        w, h = raw_image.size

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # Scale and normalize image
        image = self.img_normalize(self.standarizeImageFormat(raw_image))

        # Load depth image
        if self.use_depth:
            depth_name = os.path.join(self.DEPTH_DIR,
                                      self.depth_map[img_idx])
            raw_depth = Image.open(depth_name)

            # Scale and normalize depth image
            depth = self.normalizeDepth(raw_depth)
            sample['image'] = torch.cat((image, depth), 0)

        else:
            sample['image'] = image
            raw_depth = None

        return sample, raw_image, raw_depth

#Class to load image datasets with coco style annotations
class ImageDataset(Dataset):
    def __init__(self, cfg, coco, img_root=None, depth_root=None, data_root=None):

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.active_split = None
        self.coco = coco
        #Filter out images without object annotations
        self.coco_index = [img_id for img_id in self.coco.imgs if len(self.coco.imgToAnns[img_id])>0]
        self.n_classes = cfg.IMG_NET.N_LABELS

        #COCO doesn't have contiguous labels. These maps go from coco labels to contiguous indices and back
        self.coco_cat_map = dict(zip(range(len(self.coco.cats)), self.coco.cats.keys()))
        self.cat_coco_map = dict(zip(self.coco.cats.keys(), range(len(self.coco.cats))))
        assert len(self.coco.cats)==self.n_classes

        self.image_process = ImageProcessing(cfg, img_root, depth_root, data_root)

    def __len__(self):
        return self.length(self.active_split)

    def __getitem__(self, item):
        return self.getItem(item, self.active_split)

    def length(self, split=None):
        return len(self.coco_index)

    def getItem(self, idx, split=None):
        sample = {}

        sample['imageID'] = self.coco_index[idx]
        sample['objectIDs'] = self.coco.getAnnIds(imgIds=sample['imageID'])

        sample['objectClass'] = [self.cat_coco_map[self.coco.anns[id]['category_id']] for id in sample['objectIDs']]
        sample['class_tensor'] = torch.tensor([1 if idx in sample['objectClass'] else 0 for idx in range(self.n_classes)], dtype=torch.long, device=self.device)

        if self.image_process.use_image:
            file_name = self.coco.imgs[sample['imageID']]['file_name']
            image_features, raw_image, raw_depth = self.image_process.getAllImageFeatures(sample['imageID'], file_name)
            sample.update(image_features)

        return sample

#Class to load referring expressions datasets
class ReferExpressionDataset(Dataset):

    def __init__(self, cfg, refer, img_root=None, depth_root=None, data_root=None): #depth_dir, vocab, disable_cuda=False, transform_size=224,
                # image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], use_image=False,
                # depth_mean=19018.9, depth_std=18798.8, use_depth=False, n_contrast_object=0):

        super(ReferExpressionDataset, self).__init__()

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.image_process = ImageProcessing(cfg, img_root, depth_root, data_root)
        self.refer = refer
        self.n_contrast_object = cfg.TRAINING.N_CONSTRAST_OBJECT

        # Load the vocabulary
        with open(cfg.DATASET.VOCAB, 'r') as f:
            vocab = f.read().split()

        # Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])

        self.max_sent_len = max([len(sent['tokens']) for sent in self.refer.Sents.values()]) + 2 #For the begining and end tokens
        self.word2idx = dict(zip(vocab, range(1, len(vocab)+1)))
        self.sent2vocab(self.word2idx)

        self.index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']]
        self.train_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']
                              if self.refer.Refs[ref]['split'] == 'train']

    # Refcocog does not provide a seperate test set. So we split the val set in half.
        if cfg.DATASET.NAME == 'refcocog':
            val_labels =  self.val_index = [sent_id for ref in self.refer.Refs for sent_id in
                              self.refer.Refs[ref]['sent_ids'] if self.refer.Refs[ref]['split'] == 'val']
            n_val_labels = len(val_labels)
            self.val_index = val_labels[1:math.floor(n_val_labels/2)]
            self.test_index = val_labels[math.floor(n_val_labels/2):]
        else:
            self.val_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']
                              if self.refer.Refs[ref]['split'] == 'val']
            self.test_index = [sent_id for ref in self.refer.Refs for sent_id in self.refer.Refs[ref]['sent_ids']
                              if self.refer.Refs[ref]['split'] == 'test']

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

        sample['refID'] = self.refer.sentToRef[sent_idx]['ref_id']
        sample['imageID'] = self.refer.sentToRef[sent_idx]['image_id']
        sample['objectID'] = self.refer.sentToRef[sent_idx]['ann_id']

        if(self.refer.annToRef[sample['objectID']]['category_id'] in self.refer.Cats):
            sample['objectClass'] = self.refer.annToRef[sample['objectID']]['category_id']
        else:
            sample['objectClass'] = "unknown"

        if self.image_process.use_image or display_image:
            image_features, raw_image, raw_depth = self.getAllImageFeatures(sent_idx, display_image=display_image)
            sample.update(image_features)

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

    def getObject(self, bbox, image, depth=None):
        w, h = image.size

        pos = torch.tensor(
            [bbox[0] / w, (bbox[1] + bbox[3]) / h, (bbox[0] + bbox[2]) / w, bbox[1] / h, (bbox[2] * bbox[3]) / (w * h)],
            dtype=torch.float, device=self.device)
        rgb_object = self.image_process.img_normalize(self.image_process.standarizeImageFormat(image, bbox=bbox))

        if depth is not None:
            d_object = self.image_process.normalizeDepth(depth, bbox=bbox)
            rgbd_object = torch.cat((rgb_object, d_object), 0)
            return rgbd_object, pos
        else:
            return rgb_object, pos

    def getAllImageFeatures(self, sent_idx, display_image=False):
        ref = self.refer.sentToRef[sent_idx]

        # Load the image
        file_name = self.refer.Imgs[ref['image_id']]['file_name']
        sample, raw_image, raw_depth = self.image_process.getAllImageFeatures(ref['image_id'], file_name)

        #Extract a crop of the target bounding box
        bbox = self.refer.Anns[ref['ann_id']]['bbox']
        sample['object'], sample['pos'] = self.getObject(image=raw_image, depth=raw_depth, bbox=bbox)

        # Extract crops of contrast objects
        if self.n_contrast_object > 0:
            sample['contrast'] = self.getContrastObjects(sent_idx, n_contrast_object=self.n_contrast_object,
                                                         raw_image=raw_image, raw_depth=raw_depth)

        if display_image:
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw = ImageDraw.Draw(raw_image)
            draw.rectangle(bbox, fill=None, outline=(255, 0, 0, 255))
            del draw

        return sample, raw_image, raw_depth

    def getContrastObjects(self, sent_idx, n_contrast_object=float('inf'), raw_image=None, raw_depth=None):
        ref = self.refer.sentToRef[sent_idx]

        if raw_image is None:
            # Load the image
            file_name = self.refer.Imgs[ref['image_id']]['file_name']
            sample, raw_image, raw_depth = self.image_process.getAllImageFeatures(ref['image_id'], file_name)

        # Randomly sample all bounding boxes
        annIds = self.refer.getAnnIds(image_ids=ref['image_id'])
        bboxes = [self.refer.Anns[id]['bbox'] for id in annIds if
                  id != ref['ann_id'] and int(self.refer.Anns[id]['bbox'][2]) > 0 and int(
                      self.refer.Anns[id]['bbox'][3]) > 0]
        bboxes = random.sample(bboxes, min(n_contrast_object, len(bboxes)))

        # Get the image crops for the selected bounding boxes
        contrast = []
        for bbox in bboxes:
            object, pos = self.getObject(image=raw_image, depth=raw_depth, bbox=bbox)
            contrast.append({'object': object, 'pos': pos})

        return contrast

# TODO rewrite with config file
# Test data loader
if __name__ == "__main__":
    import argparse
    from .refer import REFER
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Test dataset loading')
    parser.add_argument('--img_root', help='path to the image directory', default='datasets/SUNRGBD/images')
    parser.add_argument('--depth_root', help='path to the image directory', default='datasets/SUNRGBD/images')
    parser.add_argument('--data_root', help='path to data directory', default='datasets/sunspot/annotations/')
    parser.add_argument('--dataset', help='dataset name', default='sunspot')
    parser.add_argument('--version', help='team that made the dataset splits', default='boulder')
    args = parser.parse_args()

    #ReferExpressionDataset test
    with open('datasets/vocab_file.txt', 'r') as f:
        vocab = f.read().split()
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    refer = REFER(data_root=args.data_root, image_dir=args.img_root, dataset=args.dataset, version=args.version)
    refer_dataset = ReferExpressionDataset(refer, args.depth_root, vocab, use_image=True, use_depth=True, n_contrast_object=1)
    sample = refer_dataset.getItem(0, display_image=True)
    # sample['raw_image'].show()

    image = refer_dataset.image_process.rev_img_normalize(sample['image'][0:3, :, :])
    image.show()
    plt.imshow(sample['image'][3, :, :])
    plt.show()

    object_img = refer_dataset.image_process.rev_img_normalize(sample['object'][0:3, :, :])
    object_img.show()
    plt.imshow(sample['object'][3, :, :])
    plt.show()

    contrast_img = refer_dataset.image_process.rev_img_normalize(sample['contrast'][0]['object'][0:3, :, :])
    contrast_img.show()
    plt.imshow(sample['contrast'][0]['object'][3, :, :])
    plt.show()

    #ImageDataset test
    from pycocotools.coco import COCO
    coco_data = COCO(os.path.join(args.data_root, 'instances.json'))
    coco_dataset = ImageDataset(coco_data, args.img_root, args.depth_root, args.data_root, use_image=True, use_depth=True)

    sample = coco_dataset.getItem(0)
    # sample['raw_image'].show()

    image = coco_dataset.image_process.rev_img_normalize(sample['image'][0:3, :, :])
    image.show()
    plt.imshow(sample['image'][3, :, :])
    plt.show()

    input("Enter to exit")