def get_weights_biases(mu, sd, input_channels, output_channels):
    import tensorflow as tf
    """
    tensorflow filter size formula for valid padding:
            Hf = H - Ho*Hs + 1
            Wf = W - Wo*Ws + 1
            Df = K
    """
    w = {
        'c1': tf.Variable(tf.truncated_normal([5, 5, input_channels, 6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([84, output_channels], mean=mu, stddev=sd)),
    }
    b = {
        'c1': tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([output_channels], mean=mu, stddev=sd))
    }
    return w, b


def convolution_layer(x, w, b, st, padding, pool_k, pool_st, dropout):
    import tensorflow as tf
    conv = tf.nn.conv2d(x, filter=w, strides=st, padding=padding)
    conv = tf.nn.bias_add(conv, bias=b)
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize=pool_k, strides=pool_st, padding=padding)
    conv = tf.nn.dropout(conv, keep_prob=dropout)
    return conv


def full_connected_layer(fc, w, b, dropout):
    import tensorflow as tf
    fc = tf.add(tf.matmul(fc, w), b)
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc, keep_prob=dropout)
    return fc


def output_layer(fc, w, b):
    import tensorflow as tf
    out = tf.add(tf.matmul(fc, w), b)
    return out


def n_parameters(layer1, layer2, layer3, layer4, layer5):
    # parameter sharing is assumed
    dim = layer1.get_shape()[3]
    layer1_params = dim * (5 * 5 * 3) + dim * 1
    dim = layer2.get_shape()[3]
    layer2_params = dim * (5 * 5 * 6) + dim * 1
    dim = layer3.get_shape()[1]
    layer3_params = dim * 400 + dim * 1
    dim = layer4.get_shape()[1]
    layer4_params = dim * 120 + dim * 1
    dim = layer5.get_shape()[1]
    layer5_params = dim * 84 + dim * 1
    total_params = layer1_params + layer2_params + layer3_params + layer4_params + layer5_params

    print("Layer 1 Params: {}".format(layer1_params))
    print("Layer 2 Params: {}".format(layer2_params))
    print("Layer 3 Params: {}".format(layer3_params))
    print("Layer 4 Params: {}".format(layer4_params))
    print("Layer 5 Params: {}".format(layer5_params))
    print("Total Params:   {}".format(total_params))


def le_net(_x_, mu, stddev, dropouts, input_channels=1, output_channels=10):
    from tensorflow.contrib.layers import flatten
    train_dropouts = {
        'c1': dropouts[0],
        'c2': dropouts[1],
        'fc1': dropouts[2],
        'fc2': dropouts[3],
    }
    w, b = get_weights_biases(mu, stddev, input_channels, output_channels)
    padding = 'VALID'
    k = 2
    st, pool_st, pool_k = [1, 1, 1, 1], [1, k, k, 1], [1, k, k, 1]
    # Layer 1 -- convolution layer:
    conv1 = convolution_layer(_x_, w['c1'], b['c1'], st, padding, pool_k, pool_st, train_dropouts['c1'])
    # Layer 2 -- convolution layer:
    conv2 = convolution_layer(conv1, w['c2'], b['c2'], st, padding, pool_k, pool_st, train_dropouts['c2'])
    # Flatten
    fc1 = flatten(conv2)
    # Layer 3 -- fully connected layer:
    fc1 = full_connected_layer(fc1, w['fc1'], b['fc1'], train_dropouts['fc1'])
    # Layer 4 -- full connected layer:
    fc2 = full_connected_layer(fc1, w['fc2'], b['fc2'], train_dropouts['fc2'])
    # Layer 5 -- fully connected output layer:
    out = output_layer(fc2, w['out'], b['out'])
    # parameters in each layer
    n_parameters(conv1, conv2, fc1, fc2, out)
    return out
