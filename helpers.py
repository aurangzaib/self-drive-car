def fetch_images_from_folder(folder, extension='*.png'):
    import cv2 as cv
    import os
    import fnmatch
    _images, _file_names = [], []
    _cwd_ = os.getcwd()
    print(_cwd_ + folder)
    for root, dir_names, file_names in os.walk(_cwd_ + folder):
        for filename in fnmatch.filter(file_names, extension):
            img = cv.imread(os.path.join(_cwd_ + folder, os.path.join(root, filename)))
            if img is not None:
                _images.append(img)
                label = filename.split('/')[-1].split('-')[0]
                _file_names.append(filename)
    return _images, _file_names


def get_classes_samples(index, labels):
    return [i for i, _x_ in enumerate(labels) if _x_ == index]


def get_new_test_data(folder):
    import numpy as np
    features, file_names = fetch_images_from_folder(folder)
    labels = [15,  # no vehicle
              19,  # Dangerous curve to the left
              29,  # Bicycles crossing
              40,  # Roundabout mandatory
              35,  # Ahead only
              26,  # Traffic signals
              26,  # Traffic signals
              40,  # Roundabout mandatory
              26,  # Traffic signals
              26,  # Traffic signals
              23,  # Slippery road
              26,  # Traffic signals
              12,  # Priority road
              12,  # Priority road
              36,  # Go straight or right
              13,  # Yield
              40  # Roundabout mandatory
              ]
    print("features length: {}, labels length: {}".format(len(features), len(labels)))
    assert (len(features) == len(labels))
    return np.array(features), labels, file_names


def traffic_sign_name(_id_):
    from pandas.io.parsers import read_csv
    sign_name = read_csv('signnames.csv').values[_id_][1]
    return sign_name


def get_batches(_batch_size_, features, labels):
    import math
    total_size, index, batch = len(features), 0, []
    n_batches = int(math.ceil(total_size / _batch_size_)) if _batch_size_ > 0 else 0
    for _i_ in range(n_batches - 1):
        batch.append([features[index:index + _batch_size_],
                      labels[index:index + _batch_size_]])
        index += _batch_size_
    batch.append([features[index:], labels[index:]])
    return batch


def load_data(filename):
    import pickle
    import os
    root = os.getcwd() + '/traffic-signs-data/'
    with open(root + filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    return data['features'], data['labels']


def save_data(filename, features, labels):
    import pickle
    root = 'traffic-signs-data/'
    assert (len(features) == len(labels))
    data = {
        'features': features,
        'labels': labels
    }
    pickle.dump(data, open(root + filename, "wb"))
    print("data saved to disk")


def grayscale(x):
    import cv2 as cv
    import numpy as np
    for index, image in enumerate(x):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        im2 = np.zeros_like(image)
        im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] = gray, gray, gray
        x[index] = im2
    return x


def normalizer(x):
    import numpy as np
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x = (x - x_min) / (x_max - x_min)
    return x


def pre_process(features, labels, is_train=False):
    from sklearn.utils import shuffle
    assert (len(features) == len(labels))
    features = grayscale(features)
    features = normalizer(features)
    if is_train:
        features, labels = shuffle(features, labels)
    return features, labels


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


def perform_rotation(image, cols, rows):
    from random import randint
    import cv2
    center = (int(cols / 2), int(cols / 2))
    angle = randint(-12, 12)
    transformer = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, transformer, (cols, rows))
    return image


def perform_translation(image, cols, rows, value):
    import cv2
    import numpy as np
    transformer = np.float32([[1, 0, value], [0, 1, value]])
    image = cv2.warpAffine(image, transformer, (cols, rows))
    return image


def perform_transformation(feature, label):
    from random import randint
    transform_level = 10
    rows, cols, channels = feature.shape
    rotational_value = randint(-int(rows / transform_level), int(rows / transform_level))
    image = perform_rotation(feature, cols, rows)
    image = perform_translation(image, cols, rows, rotational_value)
    return image, label


def augment_dataset(features, labels, n_classes):
    from random import randint
    from sklearn.utils import shuffle
    import numpy as np
    transforms_per_image = 20
    iterations = 100
    augmented_features, augmented_labels = [], []
    for _i_ in range(iterations):
        for i in range(transforms_per_image):
            # get a random class from 0 to 42
            random_class = randint(0, n_classes)
            # select 10 features and labels of that class
            selected_index = get_classes_samples(random_class, labels)[random_class:random_class + 1]
            # print("index: ", selected_index)
            selected_labels = labels[selected_index]
            # perform transformation in each of the features
            for index, transform_y in zip(selected_index, selected_labels):
                # get rows and cols of the image
                transform_x = features[index]
                rows, cols, channels = transform_x.shape
                # create several transforms from a single image
                for value in range(-int(rows), int(rows), 4):
                    # perform transformations on the image
                    aug_x, aug_y = perform_transformation(transform_x, transform_y)
                    augmented_features.append(aug_x)
                    augmented_labels.append(aug_y)
    # append the results of transformations
    augmented_features, augmented_labels = shuffle(augmented_features, augmented_labels)
    augmented_features = np.array(augmented_features)
    # assertion
    assert (len(augmented_features) == len(augmented_labels))
    return augmented_features, augmented_labels


def debug_features(features, labels):
    import cv2
    for feature, label in zip(features, labels):
        cv2.imshow("image", feature)
        print("label: {}".format(label))
        cv2.waitKey()


def save_transforms_to_disk(image, label):
    import time
    import cv2
    file_name = str(label) + '-' + str(time.time()) + '.png'
    cv2.imwrite('augmentation-results/' + file_name, image)
    cv2.waitKey(2)
