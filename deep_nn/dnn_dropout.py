# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

n_input, n_classes, n_hidden_layer = 784, 10, 256  # 784 --> 28x28 image shape
mnist = input_data.read_data_sets('/tmp/tensorflow/', one_hot=True, reshape=False)
learning_rate = tf.placeholder(tf.float32)
batch_size = 128
epoch = 50
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
bias = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_classes]))
}
# probability of perceptron to keep
keep_prob = tf.placeholder(tf.float32)
# matrix of 28x28
x = tf.placeholder("float", [None, 28, 28, 1])  # --> features
y = tf.placeholder("float", [None, n_classes])  # --> labels
# reshape to vector of 1x784
x_flat = tf.reshape(x, [-1, n_input])
# testing data
validation_x = mnist.validation.images
validation_y = mnist.validation.labels.astype(np.float32)
# testing data
test_x = mnist.test.images
test_y = mnist.test.labels.astype(np.float32)
# hidden layer
hidden_layer_input = tf.add(tf.matmul(x_flat, weights['hidden_layer']), bias['hidden_layer'])
# using rectified linear unit as an activation function
hidden_layer_output = tf.nn.relu(hidden_layer_input)
# perform dropout to reduce overfit
hidden_layer_output = tf.nn.dropout(hidden_layer_output, keep_prob)
# output layer
logits = tf.add(tf.matmul(hidden_layer_output, weights['output_layer']), bias['output_layer'])
# cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
# cost
cost = tf.reduce_mean(cross_entropy)
# optimize cost using Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# correct predictions
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# init global vars
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # train the NN epoch times on the train data
    for e in range(epoch):
        n_batches = int(mnist.train.num_examples / batch_size)
        for i in range(n_batches):
            # get subsets of dataset
            # mini-batching
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                learning_rate: 0.001,
                keep_prob: 0.5,
                x: batch_x,
                y: batch_y
            })
        if e % 10 == 0:
            validation_accuracy = sess.run(accuracy, feed_dict={
                x: validation_x,
                y: validation_y,
                keep_prob: 1.0  # always 1 in validation
            })
            print("{:2d}th epoch accuracy: {:2.3f}%".format(e, validation_accuracy * 100))
    # test the NN on test data
    test_accuracy = sess.run(accuracy, feed_dict={
        x: test_x,
        y: test_y,
        keep_prob: 1.0  # always 1 in test
    })
    print("test accuracy with dropout: {:2.3f}%".format(test_accuracy * 100))
