import argparse, os, re

import torch
import torchvision.models as models

torch.manual_seed(1)

from MaoEtAl_baseline import LanguagePlusImage
from TruncatedImageNetworks import TruncatedVGGorAlex
from ClassifierHelper import Classifier
from ReferExpressionDataset import ReferExpressionDataset

from refer_python3.refer import REFER

#Network Definition
class LanguagePlusImage_Contrast(LanguagePlusImage):

    def forward(self, ref, parameters):
        feats, contrast = self.image_forward(ref)

        #Input to LanguageModel
        ref['feats'] = feats
        correct_embedding = self.wordnet(ref=ref)

        contrast_embedding = []
        for vector in contrast:
            ref['feats'] = vector
            contrast_embedding.append(self.wordnet(ref=ref))

        return correct_embedding, contrast_embedding

    def image_forward(self, ref):
        # Global feature
        image = ref['image']
        if self.use_cuda:
            image = image.cuda()
        image_out = self.imagenet(image)

        # Object feature
        object = ref['object']
        if self.use_cuda:
            object = object.cuda()
        object_out = self.imagenet(object)

        # Position features
        # [top_left_x / W, top_left_y/H, bottom_left_x/W, bottom_left_y/H, size_bbox/size_image]
        pos = ref['pos']

        # Contrast objects
        contrast = ref['contrast']
        contrast_out = []
        for item in contrast:
            contrast_item = item['object']
            if self.use_cuda:
                contrast_item = contrast_item.cuda()
            contrast_out.append(torch.cat([image_out, self.imagenet(contrast_item), item['pos']], 1))

        # Concatenate image representations
        return torch.cat([image_out, object_out, pos], 1), contrast_out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test')
    parser.add_argument('checkpoint_prefix',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

    parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/mscoco/train2014/')
    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
                        help='Size of LSTM embedding (Default:100)')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    with open('vocab_file.txt', 'r') as f:
        vocab = f.read().split()
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_cuda, use_image=True)

    checkpt_file = LanguagePlusImage.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 2005, args.dropout)
    if (os.path.isfile(checkpt_file)):
        model = LanguagePlusImage(checkpt_file=checkpt_file, vocab=vocab, use_cuda=use_cuda)
    else:
        model = LanguagePlusImage(vocab=vocab, hidden_dim=args.hidden_dim,
                              use_cuda=use_cuda, dropout=args.dropout)

    if args.mode == 'train':
        print("Start Training")
        total_loss = model.run_training(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image': True})

    if args.mode == 'test':
        print("Start Testing")
        for i in range(10, 20):
            item = refer.getItem(i, split='val', use_image=True, display_image=True)
            item['PIL'].show()
            print(model.generate("<bos>", item))
            input('Any key to continue')
