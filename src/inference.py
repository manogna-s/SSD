import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import os
import json
import numpy as np

PROJECT_DIR = '../'
GRAPH_PB_PATH = os.path.join(PROJECT_DIR, 'trained_model/frozen_inference_graph.pb')
TEST_DIR = os.path.join(PROJECT_DIR, 'data/ShelfImages/test')
RESULTS_JSON_FILE=os.path.join(PROJECT_DIR, 'img2products.json')
# Load TensorFlow graph from pb file
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(GRAPH_PB_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Create TensorFlow session and evaluate the test images
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        #Get input and output tensors
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')
        img_paths = os.listdir(TEST_DIR)
        img2products={}
        for img_path in img_paths:
            img = cv2.imread(os.path.join(TEST_DIR, img_path))
            img = cv2.resize(img, (300, 300))
            image_np_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                feed_dict={image_tensor: image_np_expanded})
            img2products[img_path]=int(num_detections[0])
        with open(RESULTS_JSON_FILE, 'w') as fp:
            json.dump(img2products, fp)
