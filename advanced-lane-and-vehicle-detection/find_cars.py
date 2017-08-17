import glob
import pickle
import time

import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle


def define_classifier(spatial_size, hist_bins, colorspace, hist_range,
                      orient, pix_per_cell, cell_per_block, hog_channel, path,
                      return_pre_trained=False,
                      return_trained=False
                      ):
    if return_pre_trained:
        data = pickle.load(open('trained-clf.p', 'rb'))
        return data["clf"], data["x_scaler"]

    imgs_cars = glob.glob(path)
    cars_files = []
    not_cars_files = []
    for img_file in imgs_cars:
        if 'image' in img_file or 'extra' in img_file:
            not_cars_files.append(img_file)
        else:
            cars_files.append(img_file)

    # features
    car_features = extract_features(cars_files, spatial_size, hist_bins, colorspace, hist_range,
                                    orient, pix_per_cell, cell_per_block, hog_channel)
    not_cars_features = extract_features(not_cars_files, spatial_size, hist_bins, colorspace, hist_range,
                                         orient, pix_per_cell, cell_per_block, hog_channel)

    # normalized features
    features = np.vstack((car_features, not_cars_features)).astype(np.float64)
    x_scaler = StandardScaler().fit(features)
    features = x_scaler.transform(features)

    # labels
    labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

    # split dataset
    features, labels = shuffle(features, labels)
    test_size = 0.0 if return_trained is True else 0.2
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=test_size,
                                                        random_state=rand_state)

    # initialize SVM with optimized params using GridSearchCV
    clf = SVC()

    # train the classifier
    t_train_start = time.time()
    clf.fit(x_train, y_train)
    t_train_end = time.time()

    pickle.dump({"clf": clf, "x_scaler": x_scaler}, open('trained-clf.p', 'wb'))
    if return_trained is True:
        print("time taken to fit : {:.2f}s".format(t_train_end - t_train_start))
        print("returning fit classifier")
        return clf, x_scaler

        # prediction using classifier
    y_predict = clf.predict(x_test)
    score = accuracy_score(y_test, y_predict)

    # accuracy check of classifier
    print("train time: {:.2f}s".format(t_train_end - t_train_start))
    print("score: {:.3f}%".format(score * 100))


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv.cvtColor(img, cv.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv.resize(img[:, :, 0], size).ravel()
    color2 = cv.resize(img[:, :, 1], size).ravel()
    color3 = cv.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def extract_features(imgs,
                     spatial_size=(32, 32),
                     hist_bins=32,
                     cspace='RGB',
                     hist_range=(0, 256),
                     orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0
                     ):
    """
    combine spatial bin, color histogram and gradient histogram features
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img_file in imgs:
        # Read in each one by one
        img = mpimg.imread(img_file)
        # apply color conversion if other than 'RGB'
        feature_image = change_cspace(img, cspace)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # img = img.astype(np.float32) / 255

        img_tosearch = img[400:656, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        hog_features = np.hstack((hog1, hog2, hog3))

        # Apply bin_spatial() to get spatial color features
        bin_features = bin_spatial(feature_image)

        # Apply color_hist() to get color histogram features
        color_hist_features = color_hist(feature_image)

        print("in extract features: hog={} spatial={} hist={}".format(len(hog_features),
                                                                      len(bin_features),
                                                                      len(color_hist_features)))
        # concatenate all 3 types of features
        feature = np.concatenate((bin_features, color_hist_features, hog_features), axis=0)

        # Append the new feature vector to the features list
        features.append(feature)

    # Return list of feature vectors
    return features


def change_cspace(img, cspace):
    feature_image = []
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv.cvtColor(img, cv.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv.cvtColor(img, cv.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv.cvtColor(img, cv.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    return feature_image


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            print("in find-cars features: hog={} spatial={} hist={}".format(len(hog_features),
                                                                            len(spatial_features),
                                                                            len(hist_features)))

            test_features = np.vstack((spatial_features, hist_features, hog_features)).astype(np.float64)
            test_features2 = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            print("test features 1: {}".format(test_features))
            print("test features 2: {}".format(test_features2))

            # Scale features and make a prediction
            test_features = X_scaler.transform(test_features)

            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                             (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


dist_pickle = pickle.load(open("trained-clf.p", "rb"))
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hog_channel = 0  # Can be 0, 1, 2, or "ALL"

img = mpimg.imread('/Users/siddiqui/Downloads/advanced-lane-detection-data/bbox-example-image.jpg')
print("image shape: {}".format(img.shape))

spatial = (32, 32)
hist_range = (0, 256)
cspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

xy_window = (96, 96)
xy_overlap = (0.5, 0.5)
with_spatial_feature = True
with_color_feature = True
with_gradient_feature = True
window_color = (0, 0, 255)
window_thickness = 3

scale = 1.5

# get the trained classifier
clf, x_scaler = define_classifier(spatial_size,
                                  hist_bins,
                                  cspace,
                                  hist_range,
                                  orient,
                                  pix_per_cell,
                                  cell_per_block,
                                  hog_channel,
                                  "/Users/siddiqui/Downloads/advanced-lane-detection-data/data-set/**/**/*.jpeg",
                                  False,
                                  True
                                  )

y_start_top = [400, 656]

out_img = find_cars(img, y_start_top[0], y_start_top[1], scale,
                    clf, x_scaler, orient, pix_per_cell, cell_per_block,
                    spatial_size,
                    hist_bins)

plt.imshow(out_img)
