import argparse, os, re

import torch
import torchvision.models as models
import torchvision.transforms as transforms

torch.manual_seed(1)

from TruncatedImageNetworks import TruncatedVGGorAlex
from LSTM import LanguageModel
from ClassifierHelper import Classifier
from ReferExpressionDataset import ReferExpressionDataset

from refer_python3.refer import REFER

#Network Definition
class LanguagePlusImage(Classifier):

    def __init__(self, checkpt_file=None, vocab=None, hidden_dim=None, dropout=0, use_cuda=False):
        super(LanguagePlusImage, self).__init__(use_cuda)

        if checkpt_file is not None:
            m = re.search('hidden(?P<hidden>\d+)_feats(?P<feats>\d+)_dropout(?P<dropout>\d+)', checkpt_file)
            self.hidden_dim = int(m.group('hidden'))
            self.feats_dim = int(m.group('feats'))
            self.dropout_p = float(m.group('dropout'))
        else:
            self.feats_dim = 2005
            self.hidden_dim = hidden_dim
            self.dropout_p = dropout

        #Text Embedding Network
        self.wordnet = LanguageModel(vocab=vocab, additional_feat=self.feats_dim, hidden_dim=self.hidden_dim,
                                     dropout=self.dropout_p, use_cuda=self.use_cuda)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(models.vgg16(pretrained=True), maxpool=True, fix_weights=range(40))

        self.to(self.device)
        if checkpt_file is not None:
            super(LanguagePlusImage, self).load_model(checkpt_file)

    def forward(self, ref, parameters):
        ref['feats'] = self.image_forward(ref)

        #Input to LanguageModel
        return self.wordnet(ref=ref)

    def image_forward(self, ref):
        # Global feature
        image = ref['image']
        image = image.unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()
        image_out = self.imagenet(image)

        # Object feature
        object = ref['object']
        object = object.unsqueeze(0)
        if self.use_cuda:
            object = object.cuda()
        object_out = self.imagenet(object)

        # Position features
        # [top_left_x / W, top_left_y/H, bottom_left_x/W, bottom_left_y/H, size_bbox/size_image]
        pos = ref['pos']

        # Concatenate image representations
        return torch.cat([image_out.view(1, -1), object_out.view(1, -1), pos.view(1, -1)], 1)

    def targets(self, instance):
        return self.wordnet.targets(instance)

    def clear_gradients(self):
        super(LanguagePlusImage, self).clear_gradients()
        self.wordnet.clear_gradients()

    @staticmethod
    def get_checkpt_file(checkpt_file, hidden_dim, feats_dim, dropout_p):
        return '{}_hidden{}_feats{}_dropout{}.mdl'.format(checkpt_file, hidden_dim, feats_dim, dropout_p)

    def checkpt_file(self, checkpt_prefix):
        return self.get_checkpt_file(checkpt_prefix, self.hidden_dim, self.feats_dim, self.dropout_p)

    def generate(self, start_word, ref):
        feats = self.image_forward(ref)
        return self.wordnet.generate(start_word, feats)

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

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_cuda, transform=transform)

    checkpt_file = LanguagePlusImage.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 2005, args.dropout)
    if (os.path.isfile(checkpt_file)):
        model = LanguagePlusImage(checkpt_file=checkpt_file, vocab=vocab, use_cuda=use_cuda)
    else:
        model = LanguagePlusImage(vocab=vocab, hidden_dim=args.hidden_dim,
                              use_cuda=use_cuda, dropout=args.dropout)

    if args.mode == 'train':
        print("Start Training")
        total_loss = model.train(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image': True})

    if args.mode == 'test':
        print("Start Testing")
        for i in range(10, 20):
            item = refer.getItem(i, split='val', use_image=True, display_image=True)
            item['PIL'].show()
            print(model.generate("<bos>", item))
            input('Any key to continue')
