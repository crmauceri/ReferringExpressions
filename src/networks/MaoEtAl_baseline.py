import torch
import torch.nn as nn
import torchvision.models as models

from .TruncatedImageNetworks import TruncatedVGGorAlex
from .LSTM import LanguageModel
from .ClassifierHelper import Classifier, SequenceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


# As described in "Generation and comprehension of unambiguous object descriptions."
# Mao, Junhua, et al.
# CVPR 2016
# Implemented by Cecilia Mauceri

#Network Definition
class LanguagePlusImage(Classifier):

    def __init__(self, cfg):
        super(LanguagePlusImage, self).__init__(cfg, loss_function = SequenceLoss(nn.CrossEntropyLoss()))

        #Text Embedding Network
        self.wordnet = LanguageModel(cfg)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(cfg, models.vgg16(pretrained=True))

        self.to(self.device)

    def forward(self, ref):
        ref['feats'] = self.image_forward(ref)

        #Input to LanguageModel
        return self.wordnet(ref=ref)

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

        # Concatenate image representations
        if image_out.size()[0]!=object_out.size()[0]:
            image_out = image_out.repeat(object_out.size()[0], 1)
        return torch.cat([image_out, object_out, pos], 1)

    def trim_batch(self, instance):
        return self.wordnet.trim_batch(instance)

    def clear_gradients(self, batch_size):
        super(LanguagePlusImage, self).clear_gradients()
        self.wordnet.clear_gradients(batch_size)

    def run_generate(self, refer_dataset, split=None):
        refer_dataset.active_split = split
        n = len(refer_dataset)
        dataloader = DataLoader(refer_dataset)

        generated_exp = [0]*len(refer_dataset)
        for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
            instances, targets = self.trim_batch(instance)
            generated_exp[k] = dict()
            generated_exp[k]['generated_sentence'] = ' '.join(self.generate("<bos>", instance=instances))
            generated_exp[k]['refID'] = instance['refID'].item()
            generated_exp[k]['imgID'] = instance['imageID'].item()
            generated_exp[k]['objID'] = instance['objectID'][0]
            generated_exp[k]['objClass'] = instance['objectClass'][0]

        return generated_exp

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
        p2 = 0.0
        average_objects = 0.0
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
                average_objects += loss.size()[0]
                sorted_loss = np.argsort(loss)
                if sorted_loss[0] == 0:
                    correct += 1.0
                    output[k]['p@1'] = 1
                if sorted_loss[0] == 0 or sorted_loss[1] == 0:
                    p2 += 1.0
                    output[k]['p@2'] = 1

                output[k]['gt_sentence'] = ' '.join([t[0] for t in instance['tokens']])
                output[k]['refID'] = instance['refID'].item()
                output[k]['imgID'] = instance['imageID'].item()
                output[k]['objID'] = instance['objectID'][0]
                output[k]['objClass'] = instance['objectClass'][0]
                output[k]['zero-shot'] = instance['zero-shot']

        print("P@1 {}".format(correct/float(k)))
        print("P@2 {}".format(p2 / float(k)))
        print("Average objects compared to {}".format(average_objects / float(k)))

        return output

    def comprehension(self, instance, bboxes, target):
        instances = {}
        label_scores = self.forward(instances)
        sum(label_scores(target))

    def test(self, instance, targets):
        return self.generate(instance=instance)