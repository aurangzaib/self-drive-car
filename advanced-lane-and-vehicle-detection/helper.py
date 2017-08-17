import glob

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def draw_boxes(input_img, bounding_boxes, color=(0, 0, 0), thick=3):
    img_pts_drawn = np.copy(input_img)
    for box in bounding_boxes:
        cv.rectangle(img_pts_drawn, box[0], box[1], color, thick)
    return img_pts_drawn


def template_matching():
    """
    1- match template in given image
    2- find min and max location
    3- find top left and bottom right coordinates for bounding box
    """
    input_img = cv.imread(
        "/Users/siddiqui/Downloads/advanced-lane-detection-data/cutouts/bbox-example-image.jpg"
    )
    img_pts_drawn = np.copy(input_img)
    templates = glob.glob("/Users/siddiqui/Downloads/advanced-lane-detection-data/cutouts/cutout*.jpg")
    bbox_list = []
    method = cv.TM_CCOEFF

    template_imgs = [cv.imread(filename) for filename in templates]

    for template_img in template_imgs:
        # match the templates
        res = cv.matchTemplate(input_img, template_img, method)
        # get the locations
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # template dimension
        width, height = template_img.shape[1], template_img.shape[0]
        # find top left and bottom right coordinates
        top_left = min_loc if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        # save in list
        bbox_list.append((top_left, bottom_right))

    # visualize the bounding boxes
    print("found locations: ", bbox_list)
    for box in bbox_list:
        top_left, bottom_right = box[0], box[1]
        cv.rectangle(img_pts_drawn, top_left, bottom_right, color=(0, 0, 255), thickness=6)
    cv.imshow("result", img_pts_drawn)
    cv.waitKey()


def plot3d(pixels, colors_rgb,
           axis_labels=list("RGB"),
           axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def get_plot3d():
    # Read a color image
    img = cv.imread("/Users/siddiqui/Downloads/advanced-lane-detection-images/31.png")

    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                          interpolation=cv.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv.cvtColor(img_small, cv.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv.cvtColor(img_small, cv.COLOR_BGR2HSV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()

    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()
