import glob
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from helper import extract_features


class Classifier:
    @staticmethod
    def get_trained_classifier(spatial_size,
                               hist_bins,
                               colorspace,
                               hist_range,
                               orient,
                               pix_per_cell,
                               cell_per_block,
                               hog_channel, path,
                               return_pre_trained=False):

        if return_pre_trained:
            data = pickle.load(open('classifier.p', 'rb'))
            return data["clf"], data["x_scaler"]

        imgs_cars = glob.glob(path)
        cars_files = []
        not_cars_files = []
        for img_file in imgs_cars:
            if 'image' in img_file or 'extra' in img_file:
                not_cars_files.append(img_file)
            else:
                cars_files.append(img_file)

        # car features
        car_features = extract_features(cars_files, spatial_size, hist_bins, colorspace, hist_range, orient,
                                        pix_per_cell, cell_per_block, hog_channel)

        # not car features
        not_cars_features = extract_features(not_cars_files, spatial_size, hist_bins, colorspace, hist_range, orient,
                                             pix_per_cell, cell_per_block, hog_channel)

        # append the feature vertically -- i.e. grow in rows with rows constant
        features = np.vstack((car_features, not_cars_features)).astype(np.float64)

        # normalize the features
        scaler = StandardScaler().fit(features)

        # features and labels
        print("features shape in classifier: {}".format(features.shape))

        features = scaler.transform(features)
        labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

        print("labels shape in classifier: {}".format(labels.shape))

        # split dataset
        features, labels = shuffle(features, labels)

        # initialize SVM with optimized params using GridSearchCV
        # best params --> kernel='rbf', C=10
        # but makes the classifier slow
        clf = SVC()

        # train the classifier
        clf.fit(features, labels)

        pickle.dump({"clf": clf, "x_scaler": scaler}, open('classifier.p', 'wb'))
        return clf, scaler
