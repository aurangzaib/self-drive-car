import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# int, float, strings, array are encapsulated in an object called Tensor

# 0D constant
A = tf.constant(123)
# 1D constant
B = tf.constant([122, 32, 3])
# 2D constant
C = tf.constant([[1, 2, 3],
                 [5, 6, 6]])

# a session is in charge of allocating CPU/GPU and remote machines
# in tensorflow, anything needs to be inside session to get executed
# function etc can be defined outside of session
# but they need to be get called from inside of session
hello_constant = tf.constant('hello world')
with tf.Session() as sess_0:
    # create a tensor session and run the evaluation
    output = sess_0.run(hello_constant)
    print(output)


def calc(_x_, _y_):
    return tf.divide(_x_, _y_)


# c = a/b - 1
a = tf.constant(10)
b = tf.constant(3)
const = tf.constant(1)
# this will not be executed until its run inside Session
c = tf.subtract(tf.divide(a, b), tf.cast(const, tf.float64))

with tf.Session() as sess_1:
    output = sess_1.run(c)
    print("a/b - 1:", output)

# tf.placeholder and feed_dict are used when using non-constants
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
with tf.Session() as sess_2:
    z = sess_2.run(calc(x, y),  # to be executed
                   feed_dict={  # data feeding
                       x: 10.,
                       y: 2.
                   })
    print(z)


def tensor_linear_function():
    n_features = 150
    n_labels = 5
    # get random weights from a normal distribution
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    bias = tf.Variable(tf.zeros(n_labels))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)


def softmax(_x_):
    using_np = np.exp(_x_) / np.sum(np.exp(_x_), axis=0)
    using_tf = tf.nn.softmax(_x_)
    return using_tf, using_np


# logits is a one-dimensional array with 3 elements
logits = np.array([
    [1., 2, 3, 6],
    [2., 4, 5, 6],
    [3., 8, 7, 6]])
# softmax will return a one-dimensional array with 3 elements
softmax(logits)

"""
understanding sum and axis:

np.sum([[0, 1], [0, 5]], axis=0) # sum elem of column
array([0, 6])
np.sum([[0, 1], [0, 5]], axis=1) # sum elem of row
array([1, 5])
"""


def cross_entropy():
    _softmax = tf.placeholder(tf.float32)
    _onehot = tf.placeholder(tf.float32)
    # find cross entropy using variables passed from session using feed_dict
    cross_ent = - tf.reduce_sum(tf.multiply(_onehot, tf.log(_softmax)))
    with tf.Session() as sess:
        print("cross entropy:", sess.run(cross_ent, feed_dict={
            _softmax: [0.7, 0.2, 0.1],
            _onehot: [1.0, 0.0, 0.0]
        }))


# cross_entropy()

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
# Import MNIST data
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)  # with one-hot-encoding
# The features are already scaled and the data is shuffled
train_features = mnist.train.images  # shape --> (55000, 784)
test_features = mnist.test.images
train_labels = mnist.train.labels.astype(np.float32)  # shape --> (55000, 10)
test_labels = mnist.test.labels.astype(np.float32)
# Weights & bias using normal distribution
weights = tf.Variable(tf.random_normal([n_input, n_classes]))  # shape --> (784, 10)
bias = tf.Variable(tf.random_normal([n_classes]))  # shape --> (10,)

# float32 requires 4 bytes, float64 requires 8 bytes
# train features & labels and weights & bias requires 174MB of RAM and can be fit in memory easily
# memory of train_features --> 55000x784x4 --> 139MB (result/(1.24x10e6)

# for large datasets either we purchase big memories or we perform mini-batching
# mini-batching --> train classifier on small random subset at a time
#               --> inefficient because Loss can't be calculated simultaneously across all data

example_features = [
    ['F11', 'F12', 'F13', 'F14'],
    ['F21', 'F22', 'F23', 'F24'],
    ['F31', 'F32', 'F33', 'F34'],
    ['F41', 'F42', 'F43', 'F44']]

example_labels = [
    ['L11', 'L12'],
    ['L21', 'L22'],
    ['L31', 'L32'],
    ['L41', 'L42']]


def batches(batch_size, features, labels):
    total_size, batch_array, index = len(features), [], 0
    n_batches = int(math.ceil(total_size / batch_size)) if batch_size > 0 else 0
    for _i_ in range(n_batches - 1):
        batch_array.append([features[index:index + batch_size],
                            labels[index:index + batch_size]])
        index += batch_size
    batch_array.append([features[index:],
                        labels[index:]])
    return batch_array


# '/tmp/tensorflow/mnist'
def batches_udacity(batch_size, features, labels):
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)

    return output_batches


pprint(batches(0, example_features, example_labels))
