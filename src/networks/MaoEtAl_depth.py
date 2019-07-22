import torchvision.models as models

from .MaoEtAl_baseline import LanguagePlusImage
from .TruncatedImageNetworks import DepthVGGorAlex

#Network Definition
class LanguagePlusDepthImage(LanguagePlusImage):

    def __init__(self, cfg):
        super(LanguagePlusDepthImage, self).__init__(cfg)

        # Use a Depth Image Embedding Network
        self.imagenet = DepthVGGorAlex(cfg, models.vgg16(pretrained=False), loss_function=None)

        self.to(self.device)
