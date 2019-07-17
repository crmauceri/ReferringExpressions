import torch
import torch.nn as nn
import numpy as np
from .ClassifierHelper import Classifier


class ImageClassifier(Classifier):
    def __init__(self, cfg):
        super(ImageClassifier, self).__init__(cfg, loss_function=MultiplePredictionLoss(cfg, nn.CrossEntropyLoss()))

    def trim_batch(self, instance):
        return instance['image'], instance['class_tensor']


class TruncatedResNet(ImageClassifier):
    def __init__(self, cfg, resnet):
        super(TruncatedResNet, self).__init__(cfg)
        self.ResNet = resnet
        self.output_dim = 4096

        self.to(self.device)

    #Forward pass ignores average pooling and fully connected layers
    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()
        x = self.ResNet.conv1(x)
        x = self.ResNet.bn1(x)
        x = self.ResNet.relu(x)
        x = self.ResNet.maxpool(x)

        x = self.ResNet.layer1(x)
        x = self.ResNet.layer2(x)
        x = self.ResNet.layer3(x)
        x = self.ResNet.layer4(x)

        return x


class TruncatedVGGorAlex(ImageClassifier):
    def __init__(self, cfg, vgg): #, maxpool=False, ignore_classification=False, fix_weights=None, checkpoint=None):
        super(TruncatedVGGorAlex, self).__init__(cfg)
        self.VGG = vgg
        self.ignore_classification = cfg.IMG_NET.IGNORE_CLASSIFICATION

        #Remove last pooling layer
        if not cfg.IMG_NET.MAXPOOL:
            self.VGG.features = nn.Sequential(*list(vgg.features.children())[:-1])
            self.output_dim = (512, 14, 14)
        else:
            self.output_dim = (512, 7, 7)

        if cfg.IMG_NET.FIX_WEIGHTS is not None:
            self.freeze(cfg.IMG_NET.FIX_WEIGHTS)

        self.to(self.device)

     # Forward pass ignores classification layers
    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        if self.ignore_classification:
            return self.VGG.features(x)
        else:
            return self.VGG(x)

    def freeze(self, fix_weights):
        child_counter = 0
        for child in self.VGG.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class DepthVGGorAlex(TruncatedVGGorAlex):
    def __init__(self, cfg, vgg):
        super(DepthVGGorAlex, self).__init__(cfg, vgg)

        depth_input_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.VGG.features = nn.Sequential(depth_input_layer, *list(self.VGG.features.children())[1:])
        self.to(self.device)

class MultiplePredictionLoss(nn.Module):
    def __init__(self, cfg, loss_function):
        super(MultiplePredictionLoss, self).__init__()
        self.loss = loss_function

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets, per_instance=False):
        # Targets is an n-hot representing multiple correct class labels
        # Randomly select a target
        target_index = torch.tensor([np.random.choice(r.cpu().nonzero()[0]) for r in targets], device=self.device)

        # Mask other targets to prevent gradient propegation
        mask = targets.clone().detach()
        mask[:, target_index] = 0
        embeddings[mask==1] = 0

        loss = self.loss(embeddings, target_index)
        return loss


# if __name__ == "__main__":
#     import argparse, os
#     parser = argparse.ArgumentParser(description='Train object detection network')
#     parser.add_argument('mode', help='train/test')
#     parser.add_argument('checkpoint_prefix',
#                         help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')
#
#     parser.add_argument('--img_root', help='path to the image directory', default='datasets/coco/images/train2014')
#     parser.add_argument('--depth_root', help='path to the image directory', default='datasets/coco/images/megadepth')
#     parser.add_argument('--data_root', help='path to data directory', default='datasets/coco/annotations/')
#     parser.add_argument('--dataset', help='dataset name', default='coco')
#     parser.add_argument('--version', help='team that made the dataset splits', default='boulder')
#     parser.add_argument('--epochs', dest='epochs', type=int, default=1,
#                         help='Number of epochs to train (Default: 1)')
#     parser.add_argument('--learningrate', dest='learningrate', type=float, default=0.001, help='Adam Optimizer Learning Rate')
#     parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
#                         help='Training batch size')
#     parser.add_argument('--DEBUG', type=bool, default=False, help="Sets random seed to fixed value")
#
#     args = parser.parse_args()
#
#     if args.DEBUG:
#         torch.manual_seed(1)
#
#     from pycocotools.coco import COCO
#     from ReferExpressionDataset import ImageDataset
#
#     coco_data = COCO(os.path.join(args.data_root, 'instances_train2014_minus_refcocog.json'))
#     coco_dataset = ImageDataset(coco_data, args.img_root, args.depth_root, args.data_root, use_image=True,
#                                 use_depth=True)
#
#     checkpt_file = DepthVGGorAlex.get_checkpt_file(args.checkpoint_prefix)
#     if (os.path.isfile(checkpt_file)):
#         print(checkpt_file)
#         model = DepthVGGorAlex(vgg=models.vgg16(pretrained=False), checkpt_file=checkpt_file)
#     else:
#         model = DepthVGGorAlex(vgg=models.vgg16(pretrained=False))
#
#     if args.mode == 'train':
#         print("Start Training")
#         total_loss = model.run_training(args.epochs, coco_dataset, args.checkpoint_prefix, parameters={'use_image': True},
#                                         learning_rate=args.learningrate, batch_size=args.batch_size)
#
#     if args.mode == 'test':
#         print("Start Testing")
#         generated_exp = model.run_classify(coco_dataset, split='test')
#
#         import DictWriter
#         with open('{}_{}_{}_generated.csv'.format(checkpt_file.replace('models', 'output'), args.dataset, model.start_epoch), 'w') as fw:
#             fieldnames = ['generated_sentence', 'refID', 'imgID', 'objID', 'objClass']
#             writer = DictWriter(fw, fieldnames=fieldnames)
#
#             writer.writeheader()
#             for exp in generated_exp:
#                 writer.writerow(exp)
#
