from tensorflow.examples.tutorials.mnist import input_data
from helpers import train_network as train
from helpers import validate_network as validate
from helpers import test_network as test
from helpers import le_net
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


def ConvNet_using_LeNet():
    save_file = './model/lenet'
    # dataset
    mnist = input_data.read_data_sets('MNIST_data/', reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    # length assertion
    assert (len(x_train) == len(y_train))
    assert (len(x_validation) == len(y_validation))
    assert (len(x_test) == len(y_test))
    # input image shape
    input_shape = x_train[0].shape[0]  # --> 28*28
    # border padding --> transforming from 28*28 to 32*32 which LeNet can process
    padding = int((32 - input_shape) / 2)
    border = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    x_train = np.pad(x_train, border, 'constant')
    x_validation = np.pad(x_validation, border, 'constant')
    x_test = np.pad(x_test, border, 'constant')
    # shuffle
    x_train, y_train = shuffle(x_train, y_train)
    # placeholders
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])  # None allows to later accept any size
    y = tf.placeholder(tf.int32, [None])
    one_hot_y = tf.one_hot(y, 10)
    # network parameters
    epochs = 10
    batch_size = 128
    learn_rate = 0.01
    # network implementation
    logits = le_net(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    cost = tf.reduce_mean(cross_entropy)  # loss operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))  # compare with ground truth
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs):
            train(sess, x_train, y_train, batch_size, optimizer, x, y)
            validate(sess, x_validation, y_validation, accuracy, x, y, e)
        saver.save(sess, save_file)
    test(x_test, y_test, accuracy, x, y, saver, save_file)


ConvNet_using_LeNet()
