from miniflow.helper_functions import get_batches
import tensorflow as tf

"""
tensorflow uses a string identifier for Variables and Operations. if identifier is not available it creates
an identifier using <Type> as name for first node and <Type>_<node_number> for subsequent nodes.

loading saved Variables directly on modified models can generate errors
this can be solved by manually giving name to each variable
"""

save_file = './train_model/finetune_train_model.ckpt'
n_input, n_classes = 3, 2

# reset and define Variables
tf.reset_default_graph()
weights = tf.Variable(tf.random_normal([n_input, n_classes]))  # , name="weight_0"
bias = tf.Variable(tf.random_normal([n_classes]))  # , name="bias_0"
saver = tf.train.Saver()
print("weights name: {}".format(weights.name))
print("bias name: {}".format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# reset and define Variables again
# note that order of Variable definition is now changed
tf.reset_default_graph()
bias = tf.Variable(tf.random_normal([n_classes]))  # , name="bias_0"
weights = tf.Variable(tf.random_normal([n_input, n_classes]))  # , name="weight_0
saver = tf.train.Saver()
print("weights name: {}".format(weights.name))
print("bias name: {}".format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # now we will get error here
    # we wont get this error if we used the same order of definition
    # error --> Assign requires shapes of both tensors to match
    # this error occurs because tf matches Variables with their names
    # this problem can be solved by manually setting the variable names
    saver.restore(sess, save_file)
