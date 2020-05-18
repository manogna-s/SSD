## Detection Network

SSD network based on MobileNetv1 feature extractor is used in this model. 
The [pretrained model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) used was trained on COCO dataset.
The input size for the network used was 300x300.

##Training configuration

* Batch size: 24
* Data augmentation:
    The following two augmentations are applied randomly to images during training:
        1. SSD Random Crop: Images are cropped randomly with IoU of the crop and an object being 
        in the range of [0.1,0.9]. The crop is retained if it contains the center of the bounding 
        box else discarded.
        2. Random horizontal flip: The images and crops generated using the above methos are randomly
        flipped horizontally.
* Anchor boxes:  6 feature maps from the network were used with one anchor box per feature map cell
    with aspect ratio of 0.66.
* Optimizer: An RMS Prop optimizer with the following parameters were used:
```
optimizer {
   rms_prop_optimizer {
     learning_rate {
       exponential_decay_learning_rate {
         initial_learning_rate: 0.00400000018999
         decay_steps: 8000
         decay_factor: 0.7
       }
     }
     momentum_optimizer_value: 0.899999976158
     decay: 0.899999976158
     epsilon: 1.0
   }
```
Refer to config/prod_det_pipeline.config for model configuration details.

 
##Q & A:
* What is the purpose of using multiple anchors per feature map cell?
Anchor boxes span different shapes of object bounding boxes possible. Aspect ratios which are 
basically the width to height ratio of a box captures the shape of a rectangular bounding box. 
While the feature map dimensions span the scale of objects a model can detect, anchors span the 
shape of objects a model can detect. Every feature map cell corresponds to a receptive field in
the input image. There can be multiple anchors at each feature map cell allowing different 
shapes of boxes in the receptive field corresponding to that feature map cell. Depending on the
dataset, analyzing the distribution of aspect ratios in the available ground truth data can help
choose the aspect ratios for the respective task.


○ Does this problem require multiple anchors? Please justify your answer.
One anchor box would suffice in this problem. The aspect ratios of ground truth bounding box 
data are in the range of about 0.5-0.8. A mean value of 0.65 can be used as aspect ratio 
defining the anchor box per cell. The variance of aspect ratios in this dataset being small, 
the object detection network would easily be able to learn to predict accurate bounding boxes.