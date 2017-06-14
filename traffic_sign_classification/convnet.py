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


def le_net(_x_, mu, stddev, dropouts, input_channels=1, output_channels=10):
    from tensorflow.contrib.layers import flatten
    import tensorflow as tf

    train_dropouts = {
        'c1': dropouts[0],
        'c2': dropouts[1],
        'fc1': dropouts[2],
        'fc2': dropouts[3],
    }
    w, b = get_weights_biases(mu, stddev, input_channels, output_channels)
    st = [1, 1, 1, 1]
    padding = 'VALID'
    k = 2
    pool_st = [1, k, k, 1]
    pool_k = [1, k, k, 1]
    # Layer 1 -- convolution layer:
    # 32x32x1 --> 28x28x6
    conv1 = tf.nn.conv2d(_x_, filter=w['c1'], strides=st, padding=padding)
    conv1 = tf.nn.bias_add(conv1, bias=b['c1'])
    conv1 = tf.nn.relu(conv1)
    # 28x28x6 --> 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=pool_k, strides=pool_st, padding=padding)
    conv1 = tf.nn.dropout(conv1, keep_prob=train_dropouts['c1'])
    # Layer 2 -- convolution layer:
    # 14x14x6 --> 10x10x16
    conv2 = tf.nn.conv2d(conv1, filter=w['c2'], strides=st, padding=padding)
    conv2 = tf.nn.bias_add(conv2, bias=b['c2'])
    conv2 = tf.nn.relu(conv2)
    # 10x10x16 --> 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=pool_k, strides=pool_st, padding=padding)
    conv2 = tf.nn.dropout(conv2, keep_prob=train_dropouts['c2'])
    # Flatten
    # 5x5x16 --> 400
    fc1 = flatten(conv2)
    # Layer 3 -- fully connected layer:
    # 400 --> 120
    fc1 = tf.add(tf.matmul(fc1, w['fc1']), b['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=train_dropouts['fc1'])
    # Layer 4 -- full connected layer:
    # 120 --> 84
    fc2 = tf.add(tf.matmul(fc1, w['fc2']), b['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=train_dropouts['fc2'])
    # Layer 5 -- fully connected layer:
    # 84 --> 10
    out = tf.add(tf.matmul(fc2, w['out']), b['out'])
    # parameters in each layer
    n_parameters(conv1, conv2, fc1, fc2, out)
    return out


def n_parameters(layer1, layer2, layer3, layer4, layer5):
    # parameter sharing is assumed
    dim = layer1.get_shape()[3]
    layer1_params = dim * (5 * 5 * 1) + dim * 1
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
