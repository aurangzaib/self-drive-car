def classify_traffic_sign():
    from traffic_sign_classification.helper import get_batches, load_data, pre_process
    from traffic_sign_classification.convnet import le_net
    from traffic_sign_classification.visualization import get_data_summary
    from sklearn.utils import shuffle
    import tensorflow as tf

    save_file = './model/lenet'
    hyper_params = {
        "mu": 0,
        "stddev": 0.1,
        "epoch": 10,
        "batch_size": 128,
        "rate": 0.001,
        "dropouts": [1., 1., .7, .3],
        "test_dropouts": [1., 1., 1., 1.]
    }
    # load data
    x_train, y_train = load_data('train.p')
    x_validation, y_validation = load_data('test.p')
    x_test, y_test = load_data('valid.p')
    # Dataset Summary & Exploration
    input_h, input_channels, n_classes, n_samples, unique_class_indices = get_data_summary(x_train,
                                                                                           x_validation,
                                                                                           x_test,
                                                                                           y_train)
    # visualize_data(x_train, n_samples, unique_class_indices)
    # pre process the datasets
    x_train, y_train = pre_process(x_train, y_train, is_train=True)
    y_validation, y_validation = pre_process(x_validation, y_validation)
    x_test, x_test = pre_process(x_test, x_test)
    # placeholders
    x = tf.placeholder(tf.float32, [None, input_h,
                                    input_h,
                                    input_channels])
    y = tf.placeholder(tf.int32, [None])
    one_hot_y = tf.one_hot(y, n_classes)
    dropouts = tf.placeholder(tf.float32, [None])
    # network implementation
    logits = le_net(x,
                    hyper_params['mu'],
                    hyper_params['stddev'],
                    dropouts,
                    input_channels,
                    n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    cost = tf.reduce_mean(cross_entropy)  # loss operation
    # using adam optimizer rather than stochastic grad descent
    # https://arxiv.org/pdf/1412.6980v7.pdf
    optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params['rate']).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(hyper_params['epoch']):
            x_train, y_train = shuffle(x_train, y_train)
            batches = get_batches(hyper_params['batch_size'], x_train, y_train)
            for batch_x, batch_y in batches:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                sess.run(optimizer, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    dropouts: hyper_params['dropouts']
                })
            validation_accuracy = sess.run(accuracy, feed_dict={
                x: x_validation,
                y: y_validation,
                dropouts: hyper_params['test_dropouts']
            })
            print("{}th epoch accuracy: {:2.3f}%".format(e, validation_accuracy * 100))
        saver.save(sess, save_file)

    with tf.Session() as sess:
        saver.restore(sess, save_file)
        test_accuracy = sess.run(accuracy, feed_dict={
            x: x_test,
            y: y_test,
            dropouts: hyper_params['test_dropouts']
        })
        print("test accuracy: {:2.3f}%".format(test_accuracy * 100))


classify_traffic_sign()
