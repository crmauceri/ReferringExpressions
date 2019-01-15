pip install -e DepthAwareCNN
pip install -e MaskRCNN
pip install -e refer_python3/refer
pip install -e refer_python3
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../..