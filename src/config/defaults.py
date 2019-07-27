from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
# Which network to run
_C.MODEL.ARCHITECTURE = "MaoEtAl_baseline"
_C.MODEL.USE_PRETRAINED = False
_C.MODEL.PRETRAINED = "" #Set this of the model name of the pretrained model to use
_C.MODEL.DISABLE_CUDA = False

#--- Output ---#
_C.OUTPUT = CN()
_C.OUTPUT.CHECKPOINT_PREFIX = 'defaults'

#--- Dataset ---#
_C.DATASET = CN()
_C.DATASET.CLASS = "ReferingExpressionDataset"
_C.DATASET.NAME = "sunspot"
_C.DATASET.VERSION = "boulder"
_C.DATASET.IMG_ROOT = 'datasets/SUNRGBD/images' # path to the image directory
_C.DATASET.IMG_VAL_ROOT = 'datasets/SUNRGBD/images' # path to the image directory
_C.DATASET.DEPTH_ROOT = 'datasets/SUNRGBD/images' # path to the depth directory
_C.DATASET.DATA_ROOT = 'datasets/sunspot/annotations/' # path to data directory
_C.DATASET.VOCAB = 'datasets/vocab_file.txt' # path to vocab file

#--- Image preprocessing variables ---#
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

#--- LSTM Variables ---#
_C.LSTM = CN()
_C.LSTM.HIDDEN = 1024 # Size of LSTM hidden layer if there is an LSTM in the network
_C.LSTM.EMBED = 1024 # Size of LSTM embedding if there is an LSTM in the network

#--- Image feature network Variables ---#
_C.IMG_NET = CN()

_C.IMG_NET.USE_CUSTOM = False # Flag for using a self-pretrained model
_C.IMG_NET.CUSTOM = "" #Set this of the model name of the pretrained model to use
_C.IMG_NET.N_LABELS = 1000 # Number of classes in imagenet

_C.IMG_NET.FEATS = 2005 # Dimensionality of the image feature
_C.IMG_NET.IGNORE_CLASSIFICATION = False # TODO Flag to skip classification layer in image network
_C.IMG_NET.FIX_WEIGHTS = list(range(40)) # Which layers to freeze weights for in image network
_C.IMG_NET.LOSS = "BCEWithLogitsLoss"
_C.IMG_NET.LOSS_WEIGHTS = [1.0]*_C.IMG_NET.N_LABELS # TODO For unbalanced datasets, weight the classes in the loss function

#--- Training hyperparameters ---#
_C.TRAINING = CN()
_C.TRAINING.N_EPOCH = 60 # Total number of training epochs
_C.TRAINING.VALIDATION_FREQ = 5 # Runs validation every n epochs
_C.TRAINING.DROPOUT = 0.0 # Dropout probability
_C.TRAINING.L2_FRACTION = 1e-5 # Adam Optimizer weight decay
_C.TRAINING.LEARNING_RATE = 0.001 # Adam Optimizer Learning Rate
_C.TRAINING.BATCH_SIZE = 16
_C.TRAINING.N_CONSTRAST_OBJECT = 0

#--- Testing hyperparameters ---#
# Specify which sets to run testing on
_C.TEST = CN()
_C.TEST.DO_TRAIN = False
_C.TEST.DO_VAL = True
_C.TEST.DO_TEST = False