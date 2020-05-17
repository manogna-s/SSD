pip3 install tensorflow==1.14
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user pillow
pip3 install --user lxml
pip3 install --user jupyter
pip3 install --user matplotlib
pip3 install --user pycocotools
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 object_detection/builders/model_builder_test.py