def get_data_summary(x_train, x_validation, x_test, y_train):
    import numpy as np
    # Number of training examples
    n_train = len(x_train)
    # Number of validation examples
    n_validation = len(x_validation)
    # Number of testing examples.
    n_test = len(x_test)
    # What's the shape of an traffic sign image?
    image_shape = x_train[0].shape
    # How many unique classes/labels there are in the dataset.
    unique_classes, unique_class_index, n_samples = np.unique(y_train,
                                                              return_index=True,
                                                              return_inverse=False,
                                                              return_counts=True)
    n_classes = len(unique_classes)
    unique_class_index = unique_class_index.tolist()
    n_samples = n_samples.tolist()
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)

    return image_shape[0], image_shape[2], n_classes, n_samples, unique_class_index  # input channels, labels


def visualize_data(x, n_samples, unique_class_indices):
    from pandas.io.parsers import read_csv
    import matplotlib.pyplot as plt
    # % matplotlib inline
    label_signs = read_csv('signnames.csv').values[:, 1]  # fetch only sign names
    for loop, index in enumerate(unique_class_indices):
        print("Class {} -- {} -- {} samples".format(loop + 1,
                                                    label_signs[loop],
                                                    n_samples[loop]))
        image = x[index].squeeze()
        plt.figure(figsize=(2, 2))
        plt.imshow(image)
        plt.show()
