import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from classifier import define_classifier
from window_search import find_cars

orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
scale = 1
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
training_samples = "/Users/siddiqui/Downloads/advanced-lane-detection-data/data-set/**/**/*.jpeg"
test_samples = '/Users/siddiqui/Downloads/advanced-lane-detection-data/binary-original-1502624640.jpg'

clf, x_scaler = define_classifier(spatial_size,
                                  hist_bins,
                                  cspace,
                                  hist_range,
                                  orient,
                                  pix_per_cell,
                                  cell_per_block,
                                  hog_channel,
                                  training_samples,
                                  return_pre_trained=False)

img = mpimg.imread(test_samples)
width, height = img.shape[1], img.shape[0]
y_start_top = [int(height / 2), height]
out_img = find_cars(img, y_start_top[0], y_start_top[1], scale,
                    clf, x_scaler, orient, pix_per_cell, cell_per_block,
                    spatial_size,
                    hist_bins)

plt.imshow(out_img)
plt.show()
