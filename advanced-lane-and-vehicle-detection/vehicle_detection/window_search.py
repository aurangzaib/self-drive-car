import cv2 as cv
import numpy as np

from helper import get_hog_features, convert_color, bin_spatial, color_hist


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img,
              y_start,
              y_stop,
              scale,
              svc,
              scaler,
              orient,
              pix_per_cell,
              cell_per_block,
              spatial_size,
              hist_bins):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[y_start:y_stop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    n_xblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    n_yblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    n_blocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    n_xsteps = (n_xblocks - n_blocks_per_window) // cells_per_step
    n_ysteps = (n_yblocks - n_blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(n_xsteps):
        for yb in range(n_ysteps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat2 = hog2[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat3 = hog3[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract the image patch
            sub_sample_img = cv.resize(ctrans_tosearch[y_top:y_top + window, x_left:x_left + window], (64, 64))

            # Get color and gradient features
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            # print("hog features shape", hog_features.shape)

            spatial_features = bin_spatial(sub_sample_img, size=spatial_size)
            hist_features = color_hist(sub_sample_img, nbins=hist_bins)

            feature = np.hstack((spatial_features, hist_features, hog_features))

            # Scale features and make a prediction
            features = scaler.transform(np.array(feature).reshape(1, -1))
            predicted_labels = svc.predict(features)

            if predicted_labels == 1:
                x_box_left = np.int(x_left * scale)
                y_top_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                cv.rectangle(draw_img, (x_box_left, y_top_draw + y_start),
                             (x_box_left + win_draw, y_top_draw + win_draw + y_start), (0, 0, 255), 6)

    return draw_img
