import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# output channels
# output depth
# filter depth
k_output = 3

# height --> rows
# width  --> columns
# input image dimensions
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

input_height = int(X.get_shape()[1])
input_width = int(X.get_shape()[2])
input_channels = int(X.get_shape()[3])

ksize = 2

# strides --> batch, input_height, input_width, input_channels
strides = [1, ksize, ksize, 1]  # 2,2 means how much to move on input image
pool_filter = [1, ksize, ksize, 1]  # 2, 2 means how many pixels of input image as filter size

# convolution filter dimensions
h = int(np.ceil(input_height / strides[1]))
w = int(np.ceil(input_width / strides[2]))
filter_height, filter_width, filter_depth = h, w, k_output

# weights and bias
filter_weights = tf.Variable(tf.truncated_normal([
    filter_height, filter_width, input_channels, filter_depth
]))
filter_bias = tf.Variable(tf.zeros(filter_depth))

# apply CNN
conv_1 = tf.nn.conv2d(X,
                      filter_weights,
                      strides=strides,
                      padding='SAME')

# add 1D bias to the layer
conv_2 = tf.nn.bias_add(conv_1, filter_bias)

# activations
conv_3 = tf.nn.relu(conv_2)

# apply max pooling:
conv_4 = tf.nn.max_pool(conv_3,
                        ksize=pool_filter,
                        strides=strides,
                        padding='SAME')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("shape input: {}".format(X.get_shape()))
    print("shape activation functions: {}".format(sess.run(conv_2).shape))
    print("shape activations: {}".format(sess.run(conv_3).shape))
    print("shape maxpool: {}".format(sess.run(conv_4).shape))

"""
max pooling reduces size of input and reduces overfitting
overfitting is reduced as # of parameters of next layers are reduced
by retaining only the max value in each filtered area
2x2 filter with 2x2 strides are commonly used
for both filter and stride, batch and channels are 1 (0th and 3rd element)

recently, pooling is out of favour because:
     -- for big data, we are more concerned about underfit
     -- `dropout` is a better regularization technique
     -- pooling results in information loss
"""
