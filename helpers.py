from tensorflow.contrib.layers import flatten
import tensorflow as tf
import math
import os

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


def LeNet(x):
    mu = 0
    sigma = 0.1
    # filter -- [5,5, input_depth, output_depth]
    filter_weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma)),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'wd1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma)),
        'wd2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
        'logits': tf.Variable(tf.truncated_normal([84, 10], mean=mu, stddev=sigma))
    }
    filter_biases = {
        'bc1': tf.Variable(tf.truncated_normal([6])),
        'bc2': tf.Variable(tf.truncated_normal([16])),
        'bd1': tf.Variable(tf.truncated_normal([120])),
        'bd2': tf.Variable(tf.truncated_normal([84])),
        'logits': tf.Variable(tf.truncated_normal([10]))
    }
    k = 2
    strides = [1, 1, 1, 1]
    pool_strides = [1, k, k, 1]
    ksize = [1, k, k, 1]
    padding = "VALID"

    # Layer 1:
    # Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = tf.nn.conv2d(x,
                         filter=filter_weights['wc1'],
                         strides=strides,
                         padding=padding)
    conv1 = tf.nn.bias_add(conv1, filter_biases['bc1'])
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1,
                           ksize=ksize,
                           strides=pool_strides,
                           padding=padding)
    print("layer1 dimensions: {}".format(conv1.get_shape()))

    # Layer 2:
    # Convolutional. Input = 14x14x6, Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1,
                         filter=filter_weights['wc2'],
                         strides=strides,
                         padding=padding)
    conv2 = tf.nn.bias_add(conv2, filter_biases['bc2'])
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2,
                           ksize=ksize,
                           strides=pool_strides,
                           padding=padding)
    print("layer2 dimensions: {}".format(conv2.get_shape()))

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(fc1, filter_weights['wd1']), filter_biases['bd1'])
    # Activation.
    fc1 = tf.nn.relu(fc1)
    print("layer3 dimensions: {}".format(fc1.get_shape()))

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, filter_weights['wd2']), filter_biases['bd2'])
    # Activation.
    fc2 = tf.nn.relu(fc2)
    print("layer4 dimensions: {}".format(fc2.get_shape()))

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2, filter_weights['logits']), filter_biases['logits'])
    print("layer5 dimensions: {}".format(logits.get_shape()))

    # Calculating Number of Parameters:
    output_channels = conv1.get_shape()[3]
    conv1_parameters = output_channels * (5 * 5 * 1) + output_channels * 1
    output_channels = conv2.get_shape()[3]
    conv2_parameters = output_channels * (5 * 5 * 6) + output_channels * 1
    output_channels = fc1.get_shape()[1]
    fc1_parameters = output_channels * 400 + output_channels * 1
    output_channels = fc2.get_shape()[1]
    fc2_parameters = output_channels * 120 + output_channels * 1
    output_channels = logits.get_shape()[1]
    logits_parameters = output_channels * 84 + output_channels * 1

    print("Parameters: Layer1:{}, Layer2:{}, Layer3:{}, Layer4:{}, Layer5:{}, Total:{}".format(
        conv1_parameters,
        conv2_parameters,
        fc1_parameters,
        fc2_parameters,
        logits_parameters,
        conv1_parameters + conv2_parameters + fc1_parameters + fc2_parameters + logits_parameters
    ))

    return logits
