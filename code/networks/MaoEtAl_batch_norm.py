import re

import torch
import torch.nn as nn
import torchvision.models as models

from .TruncatedImageNetworks import TruncatedVGGorAlex
from .LSTM_batch_norm import LanguageModel
from .ClassifierHelper import Classifier, SequenceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# As described in "Generation and comprehension of unambiguous object descriptions."
# Mao, Junhua, et al.
# CVPR 2016
# Implemented by Cecilia Mauceri

#Network Definition
class LanguagePlusImage(Classifier):

    def __init__(self, cfg):
        super(LanguagePlusImage, self).__init__(loss_function = SequenceLoss(nn.CrossEntropyLoss()))

        #Text Embedding Network
        self.wordnet = LanguageModel(cfg)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(cfg, models.vgg16(pretrained=True))

        #Batch norm layer
        self.img1_batchnorm = nn.BatchNorm2d(3)
        self.img2_batchnorm = nn.BatchNorm1d(cfg.IMG_NET.FEATS)

        self.to(self.device)

    def forward(self, ref, parameters):
        ref['feats'] = self.image_forward(ref)

        #Input to LanguageModel
        return self.wordnet(ref=ref)

    def image_forward(self, ref):
        # Global feature
        image = ref['image']
        if self.use_cuda:
            image = image.cuda()
        image = self.img1_batchnorm(image)
        image_out = self.imagenet(image)

        # Object feature
        object = ref['object']
        if self.use_cuda:
            object = object.cuda()
        object = self.img1_batchnorm(object)
        object_out = self.imagenet(object)

        # Position features
        # [top_left_x / W, top_left_y/H, bottom_left_x/W, bottom_left_y/H, size_bbox/size_image]
        pos = ref['pos']

        # Concatenate image representations
        if image_out.size()[0]!=object_out.size()[0]:
            image_out = image_out.repeat(object_out.size()[0], 1)
        feat = torch.cat([image_out, object_out, pos], 1)

        return self.img2_batchnorm(feat)

    def trim_batch(self, instance):
        return self.wordnet.trim_batch(instance)

    def clear_gradients(self, batch_size):
        super(LanguagePlusImage, self).clear_gradients()
        self.wordnet.clear_gradients(batch_size)

    def generate(self, start_word, instance=None, feats=None):
        with torch.no_grad():
            if feats is None:
                feats = self.image_forward(instance)
            return self.wordnet.generate(start_word, feats=feats)

    def run_comprehension(self, refer_dataset, split=None, parameters=None):
        loss_fcn = SequenceLoss(nn.CrossEntropyLoss(reduce=False))
        self.eval()
        refer_dataset.active_split = split
        dataloader = DataLoader(refer_dataset, batch_size=1)

        correct = 0.0
        output = [0]*len(refer_dataset)
        for k, instance in enumerate(tqdm(dataloader, desc='Validation')):
            with torch.no_grad():
                for object in instance['contrast']:
                    for key, value in object.items():
                        instance[key] = torch.cat([instance[key], value], 0)
                del instance['contrast']
                instances, targets = self.trim_batch(instance)
                self.clear_gradients(batch_size=instances['object'].size()[0])

                output[k] = dict()
                label_scores = self(instances, parameters)
                loss = loss_fcn(label_scores, targets.repeat(label_scores.size()[0], 1), per_instance=True)
                if torch.argmin(loss).item() == 1:
                    correct += 1.0
                    output[k]['correct'] = 1

                output[k]['gt_sentence'] = ' '.join([t[0] for t in instance['tokens']])
                output[k]['refID'] = instance['refID'].item()
                output[k]['imgID'] = instance['imageID'].item()
                output[k]['objID'] = instance['objectID'].item()
                output[k]['objClass'] = instance['objectClass'][0]
                output[k]['zero-shot'] = instance['zero-shot']

        return correct/float(k), output

    def comprehension(self, instance, bboxes, target):
        instances = {}
        label_scores = self.forward(instances)
        sum(label_scores(target))

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
#     parser.add_argument('mode', help='train/test')
#     parser.add_argument('checkpoint_prefix',
#                         help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')
#
#     parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/mscoco/train2014/')
#     parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
#     parser.add_argument('--dataset', help='dataset name', default='refcocog')
#     parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
#     parser.add_argument('--epochs', dest='epochs', type=int, default=1,
#                         help='Number of epochs to train (Default: 1)')
#     parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
#                         help='Size of LSTM embedding (Default:100)')
#     parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')
#     parser.add_argument('--l2_fraction', dest='l2_fraction', type=float, default=1e-5, help='L2 Regularization Fraction')
#     parser.add_argument('--learningrate', dest='learningrate', type=float, default=0.001, help='Adam Optimizer Learning Rate')
#     parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
#                         help='Training batch size')
#     parser.add_argument('--DEBUG', type=bool, default=False)
#
#     args = parser.parse_args()
#
#     if args.DEBUG:
#         torch.manual_seed(1)
#
#     with open('vocab_file.txt', 'r') as f:
#         vocab = f.read().split()
#     # Add the start and end tokens
#     vocab.extend(['<bos>', '<eos>', '<unk>'])
#
#     checkpt_file = LanguagePlusImage.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 2005, args.dropout, args.l2_fraction)
#     if (os.path.isfile(checkpt_file)):
#         model = LanguagePlusImage(checkpt_file=checkpt_file, vocab=vocab)
#     else:
#         model = LanguagePlusImage(vocab=vocab, hidden_dim=args.hidden_dim, dropout=args.dropout, l2_fraction=args.l2_fraction)
#
#     if args.mode == 'train':
#         refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_image=True)
#         print("Start Training")
#         total_loss = model.run_training(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image': True},
#                                         learning_rate=args.learningrate, batch_size=args.batch_size, l2_reg_fraction=model.l2_fraction)
#     if args.mode == 'comprehend':
#         refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_image=True, n_contrast_object=float('inf'))
#         print("Start Comprehension")
#         acccuracy, output = model.run_comprehension(refer, split='val')
#         print("Accuracy {}".format(acccuracy))
#
#         with open('{}_{}_{}_comprehension.csv'.format(checkpt_file.replace('models', 'output'), args.dataset, model.start_epoch), 'w') as fw:
#             fieldnames = ['gt_sentence', 'refID', 'imgID', 'objID', 'objClass', 'correct', 'zero-shot']
#             writer = DictWriter(fw, fieldnames=fieldnames)
#
#             writer.writeheader()
#             for exp in output:
#                 writer.writerow(exp)
#
#     if args.mode == 'test':
#         refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_image=True)
#         print("Start Testing")
#         generated_exp = model.run_generate(refer, split='test_unique')
#
#         with open('{}_{}_{}_generated.csv'.format(checkpt_file.replace('models', 'output'), args.dataset, model.start_epoch), 'w') as fw:
#             fieldnames = ['generated_sentence', 'refID', 'imgID', 'objID', 'objClass']
#             writer = DictWriter(fw, fieldnames=fieldnames)
#
#             writer.writeheader()
#             for exp in generated_exp:
#                 writer.writerow(exp)
#
#         #
#         # for i in range(10, 100):
#         #     item = refer.getItem(i, split='test_unique', display_image=True)
#         #     item['PIL'].show()
#         #     print(generated_exp[i])
#         #     input('Any key to continue')
