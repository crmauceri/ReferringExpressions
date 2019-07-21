import torch
import torch.nn as nn
import torchvision.models as models

from .TruncatedImageNetworks import VGGorAlex
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
        self.imagenet = VGGorAlex(cfg, models.vgg16(pretrained=True), loss_function=None)

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

    # def run_generate(self, refer_dataset, split=None):
    #     refer_dataset.active_split = split
    #     n = len(refer_dataset)
    #     dataloader = DataLoader(refer_dataset)
    #
    #     generated_exp = [0]*len(refer_dataset)
    #     for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
    #         instances, targets = self.trim_batch(instance)
    #         generated_exp[k] = dict()
    #         generated_exp[k]['generated_sentence'] = ' '.join(self.generate("<bos>", instance=instances))
    #         generated_exp[k]['refID'] = instance['refID'].item()
    #         generated_exp[k]['imgID'] = instance['imageID'].item()
    #         generated_exp[k]['objID'] = instance['objectID'][0]
    #         generated_exp[k]['objClass'] = instance['objectClass'][0]
    #
    #     return generated_exp

    def generate(self, instance=None, feats=None):
        with torch.no_grad():
            if feats is None:
                feats = self.image_forward(instance)
            return self.wordnet.generate('<bos>', feats=feats)

    def comprehension(self, instances):
        with torch.no_grad():
            # Make new instances out of all the contrast objects
            for object in instances['contrast']:
                for key, value in object.items():
                    instances[key] = torch.cat([instances[key], value], 0)
            del instances['contrast']
            n_objects = instances['object'].size()[0]
            instances, targets = self.trim_batch(instances)
            self.clear_gradients(batch_size=n_objects)

            output = dict()
            label_scores = self(instances)
            loss = self.loss_function(label_scores, targets.repeat(label_scores.size()[0], 1), per_instance=True)

            sorted_loss = np.argsort(loss)
            if sorted_loss[0] == 0:
                output['p@1'] = 1
            else:
                output['p@1'] = 0
            if sorted_loss[0] == 0 or sorted_loss[1] == 0:
                output['p@2'] = 1
            else:
                output['p@2'] = 0

            output['n_objects'] = n_objects

            return output

    def test(self, instance, targets):
        output = dict()
        output['refID'] = instance['refID'].item()
        output['imgID'] = instance['imageID'].item()
        output['objID'] = instance['objectID'][0].item()
        output['objClass'] = instance['objectClass'][0].item()
        output['gt_sentence'] = " ".join([t[0] for t in instance['tokens']])

        output['gen_sentence'] = " ".join(self.generate(instance=instance))
        output.update(self.comprehension(instance))
        return output