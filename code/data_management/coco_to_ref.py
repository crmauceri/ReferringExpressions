import json, os, pickle
from PIL import Image
import numpy as np

with open('datasets/coco/refcocog/refs(google).p', 'rb') as f:
    refcocog = pickle.load(f)

with open('datasets/coco/annotations/instances_train2014_minus_refcocog.json', 'r') as f:
    coco = json.load(f)

coco_ref = []

i =0
for idx, ann in enumerate(coco['annotations']):
    ref = {'image_id': ann['image_id'],
           'split': 'train',
            'sentences': [],
            'file_name': coco['images'][ann['image_id']]['file_name'],
            'category_id': ann['category_id'],
            'ann_id': ann['id'],
            'sent_ids': idx,
            'ref_id': idx
            }

with open('datasets/coco/annotations/ref(boulder).json') as f:
    pickle.dump(coco_ref, f)