
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
        coco_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_train2014_minus_refcocog.json'))
        return ImageDataset(coco_data, cfg)
    else:
        raise ValueError("Dataset class not implemented")