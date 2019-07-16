
def networkFactory(cfg):
    if cfg.MODEL.ARCHITECTURE=="MaoEtAl_baseline":
        import networks.MaoEtAl_baseline
        return networks.MaoEtAl_baseline.LanguagePlusImage(cfg)
    if cfg.MODEL.ARCHITECTURE=="MaoEtAl_depth":
        import networks.MaoEtAl_depth
        return networks.MaoEtAl_depth.LanguagePlusImage(cfg)
    if cfg.MODEL.ARCHITECTURE=="MaoEtAl_full":
        import networks.MaoEtAl_full
        return networks.MaoEtAl_full.LanguagePlusImage(cfg)
    if cfg.MODEL.ARCHITECTURE=="DepthVGGorAlex":
        import networks.TruncatedImageNetworks
        return networks.TruncatedImageNetworks.DepthVGGorAlex(cfg)
    if cfg.MODEL.ARCHITECTURE=="LSTM":
        import networks.LSTM
        return networks.LSTM.LanguageModel(cfg)