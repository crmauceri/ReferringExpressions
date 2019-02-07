pip install -e DepthAwareCNN
pip install -e refer_python3/refer
pip install -e refer_python3

#Coco
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../..

#MaskRCNN
cd maskrcnn-benchmark/
python setup.py build develop
cd ..