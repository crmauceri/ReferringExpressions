from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
# Which network to run
_C.MODEL.ARCHITECTURE = "MaoEtAl_baseline"
_C.MODEL.USE_PRETRAINED = False
_C.MODEL.PRETRAINED = ""
_C.MODEL.DISABLE_CUDA = False

_C.DATASET = CN()
_C.DATASET.CLASS = "ReferingExpressionDataset"
_C.DATASET.NAME = "sunspot"
_C.DATASET.VERSION = "boulder"
_C.DATASET.IMG_ROOT = 'datasets/SUNRGBD/images' # path to the image directory
_C.DATASET.DEPTH_ROOT = 'datasets/SUNRGBD/images' # path to the depth directory
_C.DATASET.DATA_ROOT = 'datasets/sunspot/annotations/' # path to data directory
_C.DATASET.VOCAB = 'datasets/vocab_file.txt' # path to vocab file

_C.IMG_PROCESSING = CN()
_C.IMG_PROCESSING.TRANSFORM_SIZE = 224 #Input size for image network

# Statistics based on MSCOCO
_C.IMG_PROCESSING.USE_IMAGE = True
_C.IMG_PROCESSING.IMG_MEAN = [0.485, 0.456, 0.406]
_C.IMG_PROCESSING.IMG_STD = [0.229, 0.224, 0.225]

# Statistics based on SUNRGBD
_C.IMG_PROCESSING.USE_DEPTH = False
_C.IMG_PROCESSING.DEPTH_MEAN = 19018.9
_C.IMG_PROCESSING.DEPTH_STD = 18798.8

_C.LSTM = CN()
_C.LSTM.HIDDEN = 1024 # Size of LSTM hidden layer if there is an LSTM in the network
_C.LSTM.EMBED = 1024 # Size of LSTM embedding if there is an LSTM in the network

_C.IMG_NET = CN()
_C.IMG_NET.FEATS = 2005
_C.IMG_NET.MAXPOOL = False
_C.IMG_NET.IGNORE_CLASSIFICATION = False
_C.IMG_NET.FIX_WEIGHTS = list(range(40))
_C.IMG_NET.LOSS = "BCEWithLogitsLoss"
_C.IMG_NET.N_LABELS = 80 # Number of classes in mscoco

_C.TRAINING = CN()
_C.TRAINING.N_EPOCH = 60
_C.TRAINING.VALIDATION_FREQ = 5
_C.TRAINING.DROPOUT = 0.0 # Dropout probability
_C.TRAINING.L2_FRACTION = 1e-5 # L2 Regularization Fraction
_C.TRAINING.LEARNING_RATE = 0.001 # Adam Optimizer Learning Rate
_C.TRAINING.BATCH_SIZE = 16
_C.TRAINING.N_CONSTRAST_OBJECT = 0

_C.OUTPUT = CN()
_C.OUTPUT.CHECKPOINT_PREFIX = 'defaults'