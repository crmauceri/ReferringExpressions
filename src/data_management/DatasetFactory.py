
def datasetFactory(cfg):

    if cfg.DATASET.CLASS == "ReferingExpressionDataset":
        from .ReferExpressionDataset import ReferExpressionDataset
        from .refer import REFER
        refer = REFER(cfg)
        return ReferExpressionDataset(cfg, refer)
    elif cfg.DATASET.CLASS == "CocoDataset":
        from pycocotools.coco import COCO
        from .ReferExpressionDataset import ImageDataset
        import os

        train_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_train2014_minus_refcocog.json'))
        train = ImageDataset(cfg, train_data)

        test_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_valminusminival2014.json'))
        test = ImageDataset(cfg, test_data, img_root=cfg.DATASET.IMG_VAL_ROOT)

        val_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_minival2014.json'))
        val = ImageDataset(cfg, val_data, img_root=cfg.DATASET.IMG_VAL_ROOT)
        return train, test, val
    elif cfg.DATASET.CLASS == "ImageDataset":
        from pycocotools.coco import COCO
        from .ReferExpressionDataset import ImageDataset
        import os

        train_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_train.json'))
        train = ImageDataset(cfg, train_data)

        test_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_test.json'))
        test = ImageDataset(cfg, test_data, img_root=cfg.DATASET.IMG_VAL_ROOT)

        val_data = COCO(os.path.join(cfg.DATASET.DATA_ROOT, 'instances_val.json'))
        val = ImageDataset(cfg, val_data, img_root=cfg.DATASET.IMG_VAL_ROOT)
        return train, test, val
    else:
        raise ValueError("Dataset class not implemented")