#!/bin/sh

#  train_and_visualize.sh
#  
#
#  Created by Cecilia on 10/17/17.
#  


#Training
python3 code/SimpleClassifier_with_Tokens.py models/nyuv2_onehot_imageidx_train.pkl models/token_mdl_nopos_10e.tar --epochs 10 --use_tokens=0

python3 code/SimpleClassifier_with_Images.py models/nyuv2_onehot_imageidx_train.pkl nyuv2/nyu_depth_images models/image_nopos_mdl_embed_10e.tar --epochs 10 --use_tokens=0

python3 code/StackedAttentionNetwork.py models/nyuv2_onehot_imageidx_train.pkl data/nyu_depth_images models/san_mdl_nopos_10e.tar --epochs 10


#Result Analysis
python3 code/VisualizeResults_Tokens.py 1 models/nyuv2_onehot_imageidx_train.pkl data/nyu_depth_images models/token_mdl_nopos_10e.tar --use_tokens=0

python3 code/VisualizeResults_Tokens.py 2 models/nyuv2_onehot_imageidx_train.pkl data/nyu_depth_images models/image_nopos_mdl_embed_10e.tar --use_tokens=0

python3 code/VisualizeResults_Tokens.py 3 models/nyuv2_onehot_imageidx_train.pkl data/nyu_depth_images models/san_mdl_nopos_10e.tar
