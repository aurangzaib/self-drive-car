import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def conv2d(data, weight, bias, stride=1):
    data = tf.nn.conv2d(input=data,
                        filter=weight,
                        strides=[1, stride, stride, 1],
                        padding="SAME")
    data = tf.nn.bias_add(value=data, bias=bias)
    data = tf.nn.relu(data)
    return data


def maxpool2d(data, k=2):
    return tf.nn.max_pool(value=data,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding="SAME")


def conv_net(data, weight, bias, keep_prob):
    # layer 1:
    conv1 = conv2d(data, weight['wc1'], bias['bc1'])
    conv1 = maxpool2d(conv1)
    # layer 2:
    conv2 = conv2d(conv1, weight['wc2'], bias['bc2'])
    conv2 = maxpool2d(conv2)
    # fully connected layer:
    fc1 = tf.reshape(conv2, shape=[-1, weight['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weight['wd1']), bias['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    # output layer
    out = tf.add(tf.matmul(fc1, weight['out']), bias['out'])
    return out


mnist = input_data.read_data_sets('tmp/tensorflow/mnist', one_hot=True, reshape=False)
n_classes = 10
dropout = 0.75

learn_rate = 0.001
batch_size = 512
epochs = 10
test_validation_size = 256

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

logits = conv_net(data=x, weight=weights, bias=biases, keep_prob=dropout)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
n_batches = int(mnist.train.num_examples / batch_size)
with tf.Session() as sess:
    sess.run(init)
    # optimize parameters --> weights, biases and cost

    for e in range(epochs):
        for i in range(n_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y
            })
        if e % 10 == 0:
            loss = sess.run(cost, feed_dict={
                x: batch_x, y: batch_y
            })
            validation_accuracy = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_validation_size],
                y: mnist.validation.labels[:test_validation_size]
            })
    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_validation_size],
        y: mnist.test.labels[:test_validation_size]
    })
    print("test accuracy: {}".format(test_accuracy))
