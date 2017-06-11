from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from miniflow.helper_functions import get_batches


def epoch_stats(epoch_index, session, _features_, _labels_):
    # _features_, _labels_ --> features and labels of the current epoch

    # finding cost on the current features and labels
    current_cost = session.run(cost, feed_dict={
        features: _features_,
        labels: _labels_
    })
    # finding accuracy on the validation dataset
    validation_accuracy = session.run(accuracy, feed_dict={
        features: validation_features,
        labels: validation_labels
    })
    print("{:2d}th epoch --> cost: {:5.3f} accuracy: {:2.3f}%".format(epoch_index,
                                                                      current_cost,
                                                                      validation_accuracy * 100))
    return current_cost, validation_accuracy


def test_stats(_features_, _labels_):
    test_accuracy = sess.run(accuracy, feed_dict={
        features: _features_,
        labels: _labels_
    })
    print("test accuracy: {:2.3f}%".format(test_accuracy * 100))
    return test_accuracy


# input and class size
n_input, n_classes = 784, 10  # read mnist data
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True)
# features
train_features = mnist.train.images
validation_features = mnist.validation.images
test_features = mnist.test.images
# labels
train_labels = mnist.train.labels.astype(np.float32)
validation_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)
# placeholder for features and labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
learning_rate = tf.placeholder(tf.float32)
# weights and bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
# logits
logits = tf.add(tf.matmul(features, weights), bias)
# cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# cost
cost = tf.reduce_mean(cross_entropy)
# implement loss optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# implement accurate_prediction and accuracy
# accurate prediction is a comparison ...
# ...of logits(predicted output) and labels (actual output)
# argmax returns index of largest value
prediction_accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_accuracy, tf.float32))
# batch size and learn rate and epoch
batch_size, learn_rate, epoch = 128, 0.1, 100
# get batches of train data
train_batch = get_batches(batch_size, train_features, train_labels)
n_batches = int(mnist.train.num_examples / batch_size)
# initialize global variable
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # initialize global variable
    sess.run(init)
    # for each epoch:
    #    optimize the network cost
    #    find cost on current features and labels
    #    find the accuracy on validation data
    # after all epochs:
    #    find accuracy on test data
    for e in range(epoch):
        for i in range(n_batches):
            x, y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                learning_rate: learn_rate, features: x, labels: y
            })
        if e % 10 == 0:
            epoch_stats(e, sess, x, y)
            # test data is hidden during training and only validation data is used to...
            # ...test accuracy and cost during training
    test_stats(test_features, test_labels)

"""
using get_batches() instead of mnist builtin next_batch():
    batches = get_batches(batch_size, train_features, train_labels):
    for f, l in batches:
        sess.run(optimizer, feed_dict={...} 
"""
