import cv2 as cv
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv.cvtColor(img, cv.COLOR_RGB2LUV)


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
        """
        Uncomment the following line if you extracted training
        data from .png images (scaled 0 to 1 by mpimg) and the
        image you are searching is a .jpg (scaled 0 to 255)
        """
        # img = img.astype(np.float32) / 255

        # get hog features for either specific channel or for all channels
        if hog_channel == 'ALL':
            hog_features = []
            # get features for all 3 channels
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     feature_vec=True, vis=False))
            hog_features = np.ravel(hog_features)
        else:
            # get features for specific channel
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        bin_features = bin_spatial(feature_image, spatial_size)

        # Apply color_hist() to get color histogram features
        color_hist_features = color_hist(feature_image, hist_bins)

        # concatenate all 3 types of features
        feature = np.concatenate((bin_features, color_hist_features, hog_features), axis=0)

        # Append the new feature vector to the features list
        features.append(feature)

    # Return list of feature vectors
    return features


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
