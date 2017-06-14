from sklearn.utils import shuffle

import math
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_batches(_batch_size_, _features_, _labels_):
    total_size, index, batch = len(_features_), 0, []
    n_batches = int(math.ceil(total_size / _batch_size_)) if _batch_size_ > 0 else 0
    for _i_ in range(n_batches - 1):
        batch.append([_features_[index:index + _batch_size_],
                      _labels_[index:index + _batch_size_]])
        index += _batch_size_
    batch.append([_features_[index:], _labels_[index:]])
    return batch


def load_data(filename):
    import pickle
    root = 'traffic-signs-data/'
    with open(root + filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    return data['features'], data['labels']


def get_data_summary(x_train, x_validation, x_test, y_train):
    # Number of training examples
    n_train = len(x_train)

    # Number of validation examples
    n_validation = len(x_validation)

    # Number of testing examples.
    n_test = len(x_test)

    # What's the shape of an traffic sign image?
    image_shape = x_train[0].shape

    # How many unique classes/labels there are in the dataset.
    unique_classes, unique_class_index, n_samples = np.unique(y_train,
                                                              return_index=True,
                                                              return_inverse=False,
                                                              return_counts=True)
    n_classes = len(unique_classes)
    unique_class_index = unique_class_index.tolist()
    n_samples = n_samples.tolist()
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)

    return image_shape[0], image_shape[2], n_classes, n_samples, unique_class_index  # input channels, labels


def visualize_data(x, n_samples, unique_class_indices):
    from pandas.io.parsers import read_csv
    import numpy as np
    import matplotlib.pyplot as plt
    # % matplotlib inline
    label_signs = read_csv('signnames.csv').values[:, 1]  # fetch only sign names
    for loop, index in enumerate(unique_class_indices):
        print("Class {} -- {} -- {} samples".format(loop + 1,
                                                    label_signs[loop],
                                                    n_samples[loop]))
        image = x[index].squeeze()
        plt.figure(figsize=(2, 2))
        plt.imshow(image)
        plt.show()


def grayscale(x):
    import cv2 as cv
    for index, image in enumerate(x):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        im2 = np.zeros_like(image)
        im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] = gray, gray, gray
        x[index] = im2
    return x


def normalizer(x):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    print("original dimensions: {}, original mean: {}, original std: {}".format(x.shape,
                                                                                np.mean(x),
                                                                                np.std(x)))
    x = (x - x_min) / (x_max - x_min)
    print("normalized dimensions: {}, norm mean: {}, original std: {}".format(x.shape,
                                                                              np.mean(x),
                                                                              np.std(x)))
    return x


def pre_process(features, labels, is_train=False):
    features = grayscale(features)
    features = normalizer(features)
    if is_train:
        features, labels = shuffle(features, labels)
    return features, labels
