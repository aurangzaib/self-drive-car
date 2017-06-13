from helpers import *
import tensorflow as tf


def classify_traffic_sign():
    save_file = './model/lenet'
    hyper_params = {
        "epoch": 10,
        "batch_size": 128,
        "rate": 0.001,
        "dropout": 0.3
    }
    # load data
    x_train, y_train = load_data('/train.p')
    x_validation, y_validation = load_data('/test.p')
    x_test, y_test = load_data('/valid.p')

    # Dataset Summary & Exploration
    input_h, input_channels, n_classes = get_data_summary(x_train, x_validation, x_test)

    # pre-process dataset
    x_train, y_train, x_validation, x_test = pre_process(x_train,
                                                         y_train,
                                                         x_validation,
                                                         x_test)
    # visualize_data(x_train, y_train)
    # placeholders
    x = tf.placeholder(tf.float32, [None, input_h, input_h, input_channels])
    y = tf.placeholder(tf.int32, [None])
    one_hot_y = tf.one_hot(y, n_classes)
    dropout = tf.placeholder(tf.float32)
    # network implementation
    logits = le_net(x, input_channels, n_classes, dropout)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    cost = tf.reduce_mean(cross_entropy)  # loss operation
    optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params['rate']).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(hyper_params['epoch']):
            train_network(sess, x_train, y_train, dropout, hyper_params['batch_size'], optimizer, x, y)
            validate_network(sess, x_validation, y_validation, dropout, accuracy, x, y, e)
        saver.save(sess, save_file)
    test_network(x_test, y_test, dropout, accuracy, x, y, saver, save_file)


classify_traffic_sign()
