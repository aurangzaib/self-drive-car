from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# remove previous tensors and operations
tf.reset_default_graph()

# save directory of trained models
save_file = './train_model/train_model.ckpt'
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
features, labels = tf.placeholder(tf.float32, [None, n_input]), tf.placeholder(tf.float32, [None, n_classes])
learning_rate = tf.placeholder(tf.float32)
# weights and bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
# logits
logits = tf.add(tf.matmul(features, weights), bias)
# loss (or cost)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# cost
cost = tf.reduce_mean(cross_entropy)
# implement loss optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# implement accurate_prediction and accuracy
# accurate prediction is comparison
# of logits(predicted output) and labels (actual output)
prediction_accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_accuracy, tf.float32))
# initialize global variable
init = tf.global_variables_initializer()
# instance of Saver class
saver = tf.train.Saver()
# batch size and learn rate and epoch
batch_size, learn_rate, epoch = 128, 0.1, 100
# get batches of train data
n_batches = int(mnist.train.num_examples / batch_size)
with tf.Session() as sess:
    # initialize global variable
    sess.run(init)
    # optimize loss
    for e in range(epoch):
        for i in range(n_batches):
            x, y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                learning_rate: learn_rate,
                features: x,
                labels: y
            })
        # find accuracy
        validation_accuracy = sess.run(accuracy, feed_dict={
            features: validation_features,
            labels: validation_labels
        })
        # print every 10th accuracy
        if e % 20 == 0:
            print("{:2d}th epoch accuracy: {:2.3f}%".format(e, validation_accuracy * 100))
    # save the trained model
    saver.save(sess, save_file)

with tf.Session() as sess:
    # without restoring the trained model from disk
    # note the low accuracy
    sess.run(init)
    test_accuracy = sess.run(accuracy, feed_dict={
        features: test_features,
        labels: test_labels
    })
    print("accuracy w/o restore: {:2.3f}%".format(test_accuracy * 100))

with tf.Session() as sess:
    # restore the optimized weights, bias and model
    # note the high accuracy
    saver.restore(sess, save_file)
    test_accuracy = sess.run(accuracy, feed_dict={
        features: test_features,
        labels: test_labels
    })
    print("accuracy w/t restore: {:2.3f}%".format(test_accuracy * 100))
