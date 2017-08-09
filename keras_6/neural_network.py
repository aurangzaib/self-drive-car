from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pickle
import os

tf.python.control_flow_ops = tf
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('small_traffic_set/small_train_traffic.p', mode='rb') as filename:
    data = pickle.load(filename)
x_train, y_train = data['features'], data['labels']
x_train, y_train = shuffle(x_train, y_train)

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

xmax = float(np.max(x_train))
xmin = float(np.min(x_train))
x_train_normalized = (x_train - xmin) / (xmax - xmin)
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x_train_normalized, y=y_one_hot, epochs=30, verbose=0)

with open('small_traffic_set/small_test_traffic.p', mode='rb') as filename:
    data = pickle.load(filename)

x_test, y_test = data['features'], data['labels']
xmax = float(np.max(x_test))
xmin = float(np.min(x_test))
x_test_normalized = (x_test - xmin) / (xmax - xmin)
label_binarizer = LabelBinarizer()
y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(x=x_test_normalized, y=y_one_hot_test, verbose=0)
for index, name in enumerate(model.metrics_names):
    metric_name = name
    metric_value = metrics[index]
    print('{}: {}'.format(metric_name, metric_value))
