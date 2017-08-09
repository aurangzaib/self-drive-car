def classify_traffic_sign():
    from helper import get_batches, load_data, pre_process
    from helper import get_new_test_data, traffic_sign_name
    from helper import augment_dataset, debug_features, save_data
    from convnet import le_net
    from visualization import get_data_summary
    from visualization import train_test_examples
    from sklearn.utils import shuffle
    import tensorflow as tf
    import numpy as np
    import cv2
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # remove previous tensors and operations
    tf.reset_default_graph()
    save_file = './model/lenet'
    hyper_params = {
        "mu": 0,
        "stddev": 0.1,
        "epoch": 10,
        "batch_size": 128,
        "rate": 0.001,
        "dropouts": [.9, .9, .6, .5],
        "test_dropouts": [1., 1., 1., 1.]
    }
    # load data
    x_train, y_train = load_data('train.p')
    x_validation, y_validation = load_data('test.p')
    x_test, y_test = load_data('valid.p')
    x_augmented, y_augmented = load_data('transforms.p')

    expand_train_data = True
    if expand_train_data:
        x_train = np.append(x_train, x_augmented, axis=0)
        y_train = np.append(y_train, y_augmented)
        x_train, y_train = shuffle(x_train, y_train)

    # Dataset Summary & Exploration
    input_h, input_channels, n_classes, n_samples = get_data_summary(x_train, y_train)
    train_test_examples(x_train, x_validation, x_test)

    perform_augmentation = False
    if perform_augmentation:
        x_augmented, y_augmented = augment_dataset(x_train, y_train, n_classes)
        save_data("transforms.p", x_augmented, y_augmented)

    # pre process the datasets
    x_train_p, y_train_p = pre_process(x_train, y_train, is_train=True)
    x_validation_p, y_validation_p = pre_process(x_validation, y_validation)
    x_test_p, y_test_p = pre_process(x_test, y_test)
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
    soft_max_prb = tf.nn.softmax(logits=logits)
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

    retrain_model = True
    if retrain_model:
        with tf.Session() as sess:
            sess.run(init)
            print("Training....")
            for e in range(hyper_params['epoch']):
                x_train_p, y_train_p = shuffle(x_train_p, y_train_p)
                batches = get_batches(hyper_params['batch_size'], x_train_p, y_train_p)
                for batch_x, batch_y in batches:
                    batch_x, batch_y = shuffle(batch_x, batch_y)
                    sess.run(optimizer, feed_dict={
                        x: batch_x, y: batch_y,
                        dropouts: hyper_params['dropouts']
                    })
                validation_accuracy = sess.run(accuracy, feed_dict={
                    x: x_validation_p, y: y_validation_p,
                    dropouts: hyper_params['test_dropouts']
                })
                print("{}th epoch - before: {:2.3f}%".format(e + 1, validation_accuracy * 100))
            # saver.save(sess, save_file)
            # test the network
            test_cost, test_accuracy = evaluate(x_test_p, y_test_p, sess)
            print("loss: {:2.3f}, accuracy: {:2.3f}%".format(test_cost, test_accuracy * 100))

    test_data = False
    if test_data:
        with tf.Session() as sess:
            # saver.restore(sess, save_file)
            test_accuracy = sess.run(accuracy, feed_dict={
                x: x_test_p,
                y: y_test_p,
                dropouts: hyper_params['test_dropouts']
            })
            print("test accuracy: {:2.3f}%".format(test_accuracy * 100))

    test_new_data = False
    if test_new_data:
        with tf.Session() as sess:
            x_test_new, y_test_new = get_new_test_data('/test-data/')
            x_test_new_p, y_test_new_p = pre_process(x_test_new, y_test_new)
            saver.restore(sess, save_file)
            predicted_logits = sess.run(accuracy, feed_dict={
                x: x_test_new_p,
                y: y_test_new_p,
                dropouts: hyper_params['test_dropouts']
            })
            prediction_probabilities = sess.run(soft_max_prb, feed_dict={
                x: x_test_new_p,
                dropouts: hyper_params['test_dropouts']
            })
            print("predicted logits: {}".format(predicted_logits))
            top_p, top_i = sess.run(tf.nn.top_k(tf.constant(prediction_probabilities), k=5))
            for index in range(len(top_p)):
                for p, i in zip(top_p[index], top_i[index]):
                    print("{}:  {:2.3f}%".format(traffic_sign_name(i), p * 100))
                print("\n\n")


classify_traffic_sign()
