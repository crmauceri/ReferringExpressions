import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from  maskrcnn_benchmark.COCODemo import COCODemo

config_file = "../pyutils/MaskRCNN/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
from pycocotools.coco import COCO
coco = COCO('../data/SUNRGBD/instances_val.json')

from cv2 import imread, imshow
image = imread("../data/" + coco.dataset['images'][0]['file_name'])
predictions = coco_demo.run_on_opencv_image(image)


from DepthAwareCNN.options.test_options import TestOptions
opt = TestOptions().parse(['--root', '../pyutils/DepthAwareCNN/DepthAwareCNN/', '--name', 'NYUv2', '--dataroot', '../data/',
            '--pretrained_model', 'checkpoints/NYUv2/model/', '--which_epoch', 'best_net_rgb'], save=False)
opt.coco = '../data/SUNRGBD/instances_val.json'

from maskrcnn_benchmark.config import cfg
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.MASK_ON = True
import predictor

coco_demo = predictor.Predictor(cfg)

from DepthAwareCNN.data.data_loader import CreateDataLoader
data_loader = CreateDataLoader(opt)  # , images=['NYU0374.jpg'], depths=['NYU0374_depth.png'])
dataset, _ = data_loader.load_data()


for i, data in enumerate(dataset):
    composite = coco_demo.compute_prediction(data['image'])
    plt.imshow("COCO detections", composite)
    break