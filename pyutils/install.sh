pip install -e DepthAwareCNN

#Coco
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../..

#MaskRCNN
cd maskrcnn-benchmark/
python setup.py build develop
cd ..