def get_data_summary(feature, label):
    import numpy as np
    # What's the shape of an traffic sign image?
    image_shape = feature[0].shape
    # How many unique classes/labels there are in the dataset.
    unique_classes, n_samples = np.unique(label,
                                          return_index=False,
                                          return_inverse=False,
                                          return_counts=True)
    n_classes = len(unique_classes)
    n_samples = n_samples.tolist()
    print("Image data shape =", image_shape)
    print("n_samples:", n_samples)
    return image_shape[0], image_shape[2], n_classes, n_samples


def train_test_examples(x_train, x_validation, x_test):
    # Number of training examples
    n_train = len(x_train)
    # Number of validation examples
    n_validation = len(x_validation)
    # Number of testing examples.
    n_test = len(x_test)
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)


def iterate_data(index, x, y, high_range, steps):
    from helper import get_classes_samples
    import matplotlib.pyplot as plt
    # % matplotlib inline
    images = get_classes_samples(index, y)
    _images_ = images[:high_range:steps] if len(images) > 100 else images
    fig, axes = plt.subplots(1, int(high_range / steps), figsize=(15, 15))
    for _index, image_index in enumerate(_images_):
        image = x[image_index].squeeze()
        axes[_index].imshow(image)
    plt.show()


def visualize_data(x, y, n_classes, n_samples, high_range=10, steps=2, show_desc=True, single_class=False):
    from pandas.io.parsers import read_csv
    label_signs = read_csv('signnames.csv').values[:, 1]  # fetch only sign names
    if single_class:
        iterate_data(n_classes, x, y, high_range, steps)
    else:
        for index in range(n_classes):
            if show_desc:
                print("Class {} -- {} -- {} samples".format(index + 1, label_signs[index], n_samples[index]))
                loopover_data(index, x, y, high_range, steps)


def visualize_test_images(x, is_single=False):
    import matplotlib.pyplot as plt
    # % matplotlib inline
    if is_single:
        x = x.squeeze()
        plt.figure(figsize=(2, 2))
        plt.imshow(x)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(15, 15))
        for index, image in enumerate(x):
            image = image.squeeze()
            axes[index].imshow(image)
    plt.show()


def visualize_augmented_features(features, images_in_row=1):
    import matplotlib.pyplot as plt
    from random import randint
    # % matplotlib inline
    fig, axes = plt.subplots(1, images_in_row, figsize=(15, 15))
    for index in range(images_in_row):
        random_index = randint(0, len(features) - 1)
        image = features[random_index].squeeze()
        axes[index].imshow(image)
    plt.show()


def histogram_data(x, n_samples, n_classes):
    import matplotlib.pyplot as plt
    width = 1 / 1.5
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Samples Distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    plt.bar(range(n_classes), n_samples, width, color="blue")
    plt.show()
