import os

def networkFactory(cfg):
    if cfg.MODEL.ARCHITECTURE=="MaoEtAl_baseline":
        import networks.MaoEtAl_baseline
        model = networks.MaoEtAl_baseline.LanguagePlusImage(cfg)
    elif cfg.MODEL.ARCHITECTURE=="MaoEtAl_depth":
        import networks.MaoEtAl_depth
        model = networks.MaoEtAl_depth.LanguagePlusImage(cfg)
    elif cfg.MODEL.ARCHITECTURE=="MaoEtAl_full":
        import networks.MaoEtAl_full
        model = networks.MaoEtAl_full.LanguagePlusImage(cfg)
    elif cfg.MODEL.ARCHITECTURE=="DepthVGGorAlex":
        import networks.TruncatedImageNetworks
        import torchvision.models as models
        model = networks.TruncatedImageNetworks.DepthVGGorAlex(cfg, vgg=models.vgg16(pretrained=False))
    elif cfg.MODEL.ARCHITECTURE=="LSTM":
        import networks.LSTM
        model = networks.LSTM.LanguageModel(cfg)
    else:
        raise ValueError("Not implemented for this architecture")

    checkpt_file = model.model_file(cfg)
    if os.path.exists(checkpt_file):
        model.load_model(checkpt_file)
    elif cfg.MODEL.USE_PRETRAINED and os.path.exists(cfg.MODEL.PRETRAINED):
        model.load_model(cfg.MODEL.PRETRAINED)

    return model