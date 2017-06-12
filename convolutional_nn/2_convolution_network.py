from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# in each layer we have to find activation function and activations
# strides = [batch, stride_height, stride_width, features]
def conv2d(data, w, b, stride=1):
    """
    find activation function and activations
    """
    # convolution
    data = tf.nn.conv2d(data, w, strides=[1, stride, stride, 1], padding='SAME')
    # add bias
    data = tf.nn.bias_add(data, b)
    # find activations
    data = tf.nn.relu(data)
    return data


def maxpool2d(data, k=2):
    """
    drop the activations randomly to reduce overfit
    """
    return tf.nn.max_pool(
        data,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME'
    )


# create a 3 layers network alternating between convolution and max pooling
# followed by fully connect and output layers
def conv_net(data, w, b, drop_out):
    # Layer 1: 28*28*1 --> 28*28*32 --> 14*14*32
    conv1 = conv2d(data, w['wc1'], b['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    # Layer 2: 14*14*32 --> 14*14*64 --> 7*7*64
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # Fully Connected Layer: 7*7*64 --> 1024
    fc1 = tf.reshape(conv2, [-1, w['wd1'].get_shape().as_list()[0]])  # reshape
    fc1 = tf.add(tf.matmul(fc1, w['wd1']), b['bd1'])  # activation function
    fc1 = tf.nn.relu(fc1)  # activations
    fc1 = tf.nn.dropout(fc1, drop_out)
    # Output Layer
    out = tf.add(tf.matmul(fc1, w['out']), b['out'])
    return out


mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True, reshape=False)
learn_rate = 0.00001
epoch = 10
batch_size = 128
test_validation_size = 256

# network parameters
n_classes = 10
dropout = 0.75  # prob of keep activation units
# wc --> weight convolution
# bc --> bias convolution
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # filter_h,filter_w, image_channel, filter_size
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

logits = conv_net(x, weights, biases, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size
    for e in range(epoch):
        for i in range(n_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # optimize weights and biases
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout  # less than 1 for training
            })
        if e % 10 == 0:
            # current loss
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.0  # always 1
            })
            # validation dataset accuracy
            validation_accuracy = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_validation_size],
                y: mnist.validation.labels[:test_validation_size],
                keep_prob: 1.0  # always 1 for validation
            })
            print("{:2d}th epoch accuracy: {:2.3f}%".format(e, validation_accuracy * 100))
    # test dataset accuracy
    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_validation_size],
        y: mnist.test.labels[:test_validation_size],
        keep_prob: 1.0  # always 1 for testing
    })
    print("test accuracy: {:2.3f}%".format(test_accuracy * 100))
