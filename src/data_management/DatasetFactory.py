
def datasetFactory(cfg):

    if cfg.DATASET.CLASS == "ReferingExpressionDataset":
        from .ReferExpressionDataset import ReferExpressionDataset
        from .refer import REFER
        refer = REFER(cfg)
        return ReferExpressionDataset(refer, cfg)
    elif cfg.DATASET.CLASS == "ImageDataset":
        from pycocotools.coco import COCO
        from .ReferExpressionDataset import ImageDataset
        import os
        train = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_train2014_minus_refcocog.json'))
        test = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_valminusminival2014.json'))
        val = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_minival2014.json'))
        return ImageDataset(train, cfg), ImageDataset(test, cfg), ImageDataset(val, cfg)
    else:
        raise ValueError("Dataset class not implemented")