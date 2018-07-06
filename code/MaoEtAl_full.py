import argparse, os

import torch
import torch.nn as nn

torch.manual_seed(1)

from MaoEtAl_baseline import LanguagePlusImage
from ReferExpressionDataset import ReferExpressionDataset

from refer_python3.refer import REFER

#Network Definition
class LanguagePlusImage_Contrast(LanguagePlusImage):

    def __init__(self, checkpt_file=None, vocab=None, hidden_dim=None, dropout=0, use_cuda=False):
        super(LanguagePlusImage_Contrast, self).__init__(checkpt_file, vocab, hidden_dim, dropout, use_cuda)

        self.loss_function = MMI_softmax_Loss()

    def forward(self, ref, parameters):
        feats, contrast = self.image_forward(ref)

        #Input to LanguageModel
        ref['feats'] = feats
        embedding = self.wordnet(ref=ref)

        if self.training:
            for object in contrast:
                ref['feats'] = object
                embedding = torch.cat([embedding, self.wordnet(ref=ref)], 0)

        return embedding

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

        contrast_out = []
        if self.training:
            # Contrast objects
            for contrast in ref['contrast']:
                if self.use_cuda:
                    contrast_item = contrast['object'].cuda()
                else:
                    contrast_item = contrast['object']
                contrast_out.append(torch.cat([image_out, self.imagenet(contrast_item), contrast['pos']], 1))

        # Concatenate image representations
        return torch.cat([image_out, object_out, pos], 1), contrast_out


class MMI_MM_Loss(nn.Module):
    def __init__(self):
        super(MMI_MM_Loss, self).__init__()
        self.NNLLoss = nn.NLLLoss()

    def forward(self, embeddings, targets):
        #TODO
        dim = targets.size()[0]
        examples = embeddings[:dim, :]
        contrast = torch.zeros(examples.size())
        return self.NNLLoss(examples, targets)


class MMI_softmax_Loss(nn.Module):
    def __init__(self):
        super(MMI_softmax_Loss, self).__init__()
        self.NNLLoss = nn.NLLLoss()
        self.device = torch.device('cuda')

    def forward(self, embeddings, targets):
        #First half is positive examples, second half is contrast examples
        dim = targets.size()[0]
        examples = embeddings[:dim, :]
        contrast = torch.zeros(examples.size(), device=self.device, dtype=torch.float)
        for i in range(dim, embeddings.size()[0]):
            contrast[i % dim, :] += embeddings[i, :]
        #
        return self.NNLLoss(examples / contrast, targets)

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

    refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_cuda,
                                   use_image=True, n_contrast_object=2)

    checkpt_file = LanguagePlusImage_Contrast.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 2005, args.dropout)
    if (os.path.isfile(checkpt_file)):
        model = LanguagePlusImage_Contrast(checkpt_file=checkpt_file, vocab=vocab, use_cuda=use_cuda)
    else:
        model = LanguagePlusImage_Contrast(vocab=vocab, hidden_dim=args.hidden_dim,
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
