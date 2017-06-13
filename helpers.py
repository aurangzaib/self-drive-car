from tensorflow.contrib.layers import flatten
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from sklearn.utils import shuffle
import tensorflow as tf
import math
import os
import numpy as np
import pickle

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


"""
tensorflow filter size formula:

For Valid Padding:
        Hf = H - Ho*Hs + 1
        Wf = W - Wo*Ws + 1
        Df = K

no idea how to find in case of Same Padding
"""


def le_net(_x_, input_channels=1, output_channels=10, dropout=1.0):
    mu = 0
    sd = 0.1
    w = {
        'c1': tf.Variable(tf.truncated_normal([5, 5, input_channels, 6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([84, output_channels], mean=mu, stddev=sd)),
    }
    b = {
        'c1': tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([output_channels], mean=mu, stddev=sd))
    }
    st = [1, 1, 1, 1]
    padding = 'VALID'
    k = 2
    pool_st = [1, k, k, 1]
    pool_k = [1, k, k, 1]

    # Layer 1 -- convolution layer:
    # 32x32x1 --> 28x28x6
    conv1 = tf.nn.conv2d(_x_, filter=w['c1'], strides=st, padding=padding)
    conv1 = tf.nn.bias_add(conv1, bias=b['c1'])
    conv1 = tf.nn.relu(conv1)
    # 28x28x6 --> 14x14x6
    # conv1 = tf.nn.max_pool(conv1, ksize=pool_k, strides=pool_st, padding=padding)
    conv1 = tf.nn.dropout(conv1, keep_prob=dropout)
    # Layer 2 -- convolution layer:
    # 14x14x6 --> 10x10x16
    conv2 = tf.nn.conv2d(conv1, filter=w['c2'], strides=st, padding=padding)
    conv2 = tf.nn.bias_add(conv2, bias=b['c2'])
    conv2 = tf.nn.relu(conv2)
    # 10x10x16 --> 5x5x16
    # conv2 = tf.nn.max_pool(conv2, ksize=pool_k, strides=pool_st, padding=padding)
    conv2 = tf.nn.dropout(conv2, keep_prob=dropout)
    # Flatten
    # 5x5x16 --> 400
    fc1 = flatten(conv2)

    # Layer 3 -- fully connected layer:
    # 400 --> 120
    fc1 = tf.add(tf.matmul(fc1, w['fc1']), b['fc1'])
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
    # Layer 4 -- full connected layer:
    # 120 --> 84
    fc2 = tf.add(tf.matmul(fc1, w['fc2']), b['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout)
    # Layer 5 -- fully connected layer:
    # 84 --> 10
    out = tf.add(tf.matmul(fc2, w['out']), b['out'])
    # parameters in each layer
    n_parameters(conv1, conv2, fc1, fc2, out)
    # return convolutional neural network
    return out


def n_parameters(layer1, layer2, layer3, layer4, layer5):
    # parameter sharing is assumed
    dim = layer1.get_shape()[3]
    layer1_params = dim * (5 * 5 * 1) + dim * 1
    dim = layer2.get_shape()[3]
    layer2_params = dim * (5 * 5 * 6) + dim * 1
    dim = layer3.get_shape()[1]
    layer3_params = dim * 400 + dim * 1
    dim = layer4.get_shape()[1]
    layer4_params = dim * 120 + dim * 1
    dim = layer5.get_shape()[1]
    layer5_params = dim * 84 + dim * 1
    total_params = layer1_params + layer2_params + layer3_params + layer4_params + layer5_params

    print("Layer 1 Params: {}".format(layer1_params))
    print("Layer 2 Params: {}".format(layer2_params))
    print("Layer 3 Params: {}".format(layer3_params))
    print("Layer 4 Params: {}".format(layer4_params))
    print("Layer 5 Params: {}".format(layer5_params))
    print("Total Params:   {}".format(total_params))


def train_network(sess, x_train, y_train, dropout, batch_size, optimizer, x, y):
    x_train, y_train = shuffle(x_train, y_train)
    batches = get_batches(batch_size, x_train, y_train)
    for batch_x, batch_y in batches:
        sess.run(optimizer, feed_dict={
            x: batch_x,
            y: batch_y,
            dropout: 0.9
        })


def validate_network(sess, x_validation, y_validation, dropout, accuracy, x, y, e):
    acc = sess.run(accuracy, feed_dict={
        x: x_validation,
        y: y_validation,
        dropout: 1.0
    })
    print("{}th epoch accuracy: {:2.3f}%".format(e, acc * 100))
    return acc


def test_network(x_test, y_test, accuracy, x, y, dropout, saver, save_file):
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        acc = sess.run(accuracy, feed_dict={
            x: x_test,
            y: y_test,
            dropout: 1.0
        })
        print("test accuracy: {:2.3f}%".format(acc * 100))
        return acc


def load_data(filename):
    root = 'traffic-signs-data'
    with open(root + filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    return data['features'], data['labels']


def gray_scale(image):
    return equalize_hist(rgb2gray(image))


def get_data_summary(x_train, x_validation, x_test):
    # Number of training examples
    n_train = len(x_train)

    # Number of validation examples
    n_validation = len(x_validation)

    # Number of testing examples.
    n_test = len(x_test)

    # What's the shape of an traffic sign image?
    image_shape = x_train[0].shape

    # How many unique classes/labels there are in the dataset.
    n_classes = 43

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return image_shape[0], image_shape[2], n_classes  # input channels, labels


def visualize_data(x, y):
    import random
    import matplotlib.pyplot as plt

    index = random.randint(0, len(x))
    image = x[index].squeeze()
    image = gray_scale(image)
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.show()
    print(y[index])


def pre_process(x_train, y_train, x_validation, x_test):
    # input image shape
    input_shape = x_train[0].shape[0]
    # border padding --> transforming from 28*28 to 32*32 which LeNet can process
    padding = int((32 - input_shape) / 2)
    border = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    x_train = np.pad(x_train, border, 'constant')
    x_validation = np.pad(x_validation, border, 'constant')
    x_test = np.pad(x_test, border, 'constant')
    # shuffle
    x_train, y_train = shuffle(x_train, y_train)
    # normalize train images
    for image in x_train:
        gray_scale(image)
    return x_train, y_train, x_validation, x_test


def image_normalize(image):
    for index, pixel in enumerate(image):
        pixel[0] = (pixel[0] - 128) / 128
        pixel[1] = (pixel[1] - 128) / 128
        pixel[2] = (pixel[2] - 128) / 128
        image[index] = pixel
    return image
