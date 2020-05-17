

This page walks through the steps required to train an object detection model
on a local machine. The source code is taken from [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/8518d053936aaf30afb9ed0a4ea01baddca5bd17/research/object_detection). 

# Installation

This code is tested on an Ubuntu system. It requires Python 3 to be installed.
### Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim 
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (1.15)
*   Cython
*   contextlib2
*   cocoapi
*   opencv

To install
```bash
#Set project directory root path 
$export PROJECT_DIR=`pwd`
#From PROJECT_DIR/src
$bash setup_object_detection_api.sh
```
This installs all the library dependencies, compiling the configuration protobufs and setting up the Python
environment.


## Recommended Directory Structure for Training and Evaluation

```
+PROJECT_DIR
    +data
      -label_map file
      -train TFRecord file
      -eval TFRecord file
    +src
      + object_detection
      + slim
    +config
      -prod_det_pipeline.config
```

###Data Preparation

Download data, get train and test annotations
```bash
#From PROJECT_DIR
$bash prepare_data.sh
```

Prepare tfrecord files used for training and validation
```bash
#From PROJECT_DIR/src
$python3 object_detection/dataset_tools/create_shelf_tf_record.py
```

## Training

###Get Pretrained model(SSD MobileNetv1 trained for COCO used here)

Run the following command to download pretrained model which saves it to PROJECT_DIR/pretrained_ckpt
```bash
#From PROJECT_DIR
$bash get_pretrained_model.sh
```

Training can be initiated with the following command:

```bash
# From PROJECT_DIR/src
$bash train.sh
```
The following training config parameters can be set in train.sh 
```
# Contents of train.sh
PIPELINE_CONFIG_PATH=PROJECT_DIR/config/prod_det_pipeline.config
MODEL_DIR=PROJECT_DIR/model_logs
NUM_TRAIN_STEPS=50000
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --alsologtostderr
```
where `${PIPELINE_CONFIG_PATH}` points to the pipeline config and
`${MODEL_DIR}` points to the directory in which training checkpoints
and events will be written to. 

## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
$tensorboard --logdir=${MODEL_DIR}
```

where `${MODEL_DIR}` points to the directory that contains the
train and eval directories. Please note it may take Tensorboard a couple minutes
to populate with data.