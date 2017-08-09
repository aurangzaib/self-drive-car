from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os

tf.python.control_flow_ops = tf
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(filename):
    import pickle
    import os
    root = os.getcwd()
    with open(root + filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    return data['features'], data['labels']

X_train, y_train = load_data('/keras_6/small_traffic_set/small_train_traffic.p')
X_test, y_test = load_data('/keras_6/small_traffic_set/small_test_traffic.p')

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train,
                                                                      test_size=0.33,
                                                                      random_state=42)

X_train, y_train = shuffle(X_train, y_train)

print("train shape: {}".format(X_train.shape))
nb_classes = len(np.unique(y_train))
print("classes : {}".format(nb_classes))
model = Sequential()
# Layer 1 - convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.9))
model.add(Activation(activation='relu'))
# Layer 2 -- convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.9))
model.add(Activation(activation='relu'))
# Flatten
model.add(Flatten(input_shape=(32, 32, 3)))
# Layer 3 -- fully connected layer
model.add(Dense(128))
model.add(Dropout(rate=0.6))
model.add(Activation(activation='relu'))
# Layer 4 -- output layer
model.add(Dense(5))
model.add(Activation(activation='softmax'))

xmax = float(np.max(X_train))
xmin = float(np.min(X_train))
x_train_normalized = (X_train - xmin) / (xmax - xmin)
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x_train_normalized, y=y_one_hot, epochs=30, verbose=0)

xmax = float(np.max(X_test))
xmin = float(np.min(X_test))
x_test_normalized = (X_test - xmin) / (xmax - xmin)
label_binarizer = LabelBinarizer()
y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(x=x_test_normalized, y=y_one_hot_test, verbose=0)
for index, name in enumerate(model.metrics_names):
    metric_name = name
    metric_value = metrics[index]
    print('{}: {}'.format(metric_name, metric_value))
