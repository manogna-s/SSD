import os
import cv2
import json


#Splt annotations into train and test based on image split
def get_train_test_annotations(data_dir, annotations_txt, train_annotations, test_annotations):
    train_dir = os.path.join(data_dir, 'train')
    train_filenames = os.listdir(train_dir)
    test_dir = os.path.join(data_dir, 'test')
    test_filenames = os.listdir(test_dir)
    train_annotations_txt = open(train_annotations, 'w')
    test_annotations_txt = open(test_annotations, 'w')
    annotations = open(annotations_txt).readlines()
    for annotation in annotations:
        file_name = annotation.split(' ')[0]
        if file_name in train_filenames:
            train_annotations_txt.write(annotation)
        elif file_name in test_filenames:
            test_annotations_txt.write(annotation)


data_dir = './ShelfImages'
annotation_file = './annotations/annotation.txt'
train_annotations = os.path.join(data_dir, 'train_annotations.txt')
test_annotations = os.path.join(data_dir, 'test_annotations.txt')
get_train_test_annotations(data_dir, annotation_file, train_annotations, test_annotations)


