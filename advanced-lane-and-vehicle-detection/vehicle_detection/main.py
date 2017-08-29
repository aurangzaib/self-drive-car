import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

from classifier import Classifier
from helper import add_heat, apply_threshold, draw_labeled_bboxes
from window_search import WindowSearch

orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
scale = 1.5
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
training = "/Users/siddiqui/Downloads/advanced-lane-detection-data/data-set/**/**/*.jpeg"
testing = '/Users/siddiqui/Documents/Projects/self-drive/CarND-Advanced-Lane-Lines/buffer/binary-original-1502624640.jpg'

clf, x_scaler = Classifier.get_trained_classifier(spatial_size,
                                                  hist_bins,
                                                  cspace,
                                                  hist_range,
                                                  orient,
                                                  pix_per_cell,
                                                  cell_per_block,
                                                  hog_channel,
                                                  training,
                                                  return_pre_trained=True)

imgs = glob.glob(testing)
for filename in imgs:
    img = mpimg.imread(filename)
    # 3 channel without alpha
    img = img[:, :, :3]
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    width, height = img.shape[1], img.shape[0]
    y_start_top = [int(height / 2), height]

    bounding_boxes = WindowSearch.get_bounding_boxes(img, y_start_top[0], y_start_top[1],
                                                     scale, clf, x_scaler, orient, pix_per_cell,
                                                     cell_per_block, spatial_size, hist_bins)

    # Add heat to each box in box list
    heat = add_heat(heat, bounding_boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    plt.imshow(draw_img)
    plt.pause(0.000001)
plt.show()
