import torch
import torch.nn as nn

from .MaoEtAl_baseline import LanguagePlusImage

#Network Definition
class LanguagePlusBatchImage(LanguagePlusImage):

    def __init__(self, cfg):
        super(LanguagePlusBatchImage, self).__init__(cfg)

        #Batch norm layer
        self.img1_batchnorm = nn.BatchNorm2d(3)
        self.img2_batchnorm = nn.BatchNorm1d(cfg.IMG_NET.FEATS)

        self.to(self.device)

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
