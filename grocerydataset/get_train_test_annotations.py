import os


def get_train_test_annotations(data_dir, annotations_txt):
    train_dir = os.path.join(data_dir, 'train')
    train_filenames = os.listdir(train_dir)
    test_dir = os.path.join(data_dir, 'test')
    test_filenames = os.listdir(test_dir)
    train_annotations_txt = open(os.path.join(data_dir, 'train_annotations.txt'), 'w')
    test_annotations_txt = open(os.path.join(data_dir, 'test_annotations.txt'), 'w')
    annotations = open(annotations_txt).readlines()
    for annotation in annotations:
        file_name = annotation.split(' ')[0]
        if file_name in train_filenames:
            train_annotations_txt.write(annotation)
        elif file_name in test_filenames:
            test_annotations_txt.write(annotation)


get_train_test_annotations('./ShelfImages', './annotation.txt')
