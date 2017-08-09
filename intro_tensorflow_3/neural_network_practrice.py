from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import math
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


# implement a neural network
n_input, n_classes, n_hidden_layer = 28 * 28, 10, 256
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True)

# features and labels placeholders
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
learn_rate = tf.placeholder(tf.float32)
# features and labels
train_features, train_labels = mnist.train.images, mnist.train.labels.astype(np.float32)
validation_features, validation_labels = mnist.validation.images, mnist.validation.labels.astype(np.float32)
test_features, test_labels = mnist.test.images, mnist.test.labels.astype(np.float32)

# weights and bias
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'output': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
bias = {
    'hidden': tf.Variable(tf.random_normal([n_hidden_layer])),
    'output': tf.Variable(tf.random_normal([n_classes]))
}
# hidden layer
input_hidden = tf.add(tf.matmul(features, weights['hidden']), bias['hidden'])
output_hidden = tf.nn.relu(input_hidden)

# logits
logits = tf.add(tf.matmul(output_hidden, weights['output']), bias['output'])

# probability using softmax
probabilities = tf.nn.softmax(logits=logits)

# cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

# cost
cost = tf.reduce_mean(cross_entropy)

# optimizer using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)

# prediction and accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size, epoch = 128, 100
# global variable initializer
init = tf.global_variables_initializer()

# mini-batches of train data
batches = get_batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)
    for e in range(epoch):
        for batch_features, batch_labels in batches:
            sess.run(optimizer, feed_dict={
                learn_rate: 0.1,
                features: batch_features,
                labels: batch_labels
            })
    test_accuracy = sess.run(accuracy, feed_dict={
        features: test_features,
        labels: test_labels
    })
print("accuracy: {}".format(test_accuracy))
