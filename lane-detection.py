import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv


def _first():
    image = mpimg.imread('exit-ramp.jpg')
    x_size = image.shape[1]
    y_size = image.shape[0]

    color_select = np.copy(image)
    line_image = np.copy(image)

    red_thresh = 200
    green_thresh = 200
    blue_thresh = 200
    rgb_threshold = [red_thresh, green_thresh, blue_thresh]

    left_bottom = [0, 539]
    right_bottom = [1000, 539]
    apex = [500, 0]

    # line_fitting [y = mx+b]
    # np.polyfit returns (m,b) of y = mx+b
    fit_left = np.polyfit(
        (left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit(
        (right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit(
        (left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # identify pixels below the threshold
    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (
        image[:, :, 2] < rgb_threshold[2])
    XX, YY = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & (YY > (XX * fit_right[0] + fit_right[1])) & (
        YY < (XX * fit_bottom[0] + fit_bottom[1]))

    # Mask color selection
    color_select[color_thresholds] = [0, 0, 0]
    # Find where image is both colored right and in the region
    line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

    # Display our two output images
    plt.imshow(color_select, cmap="gray")
    # plt.imshow(line_image)
    plt.show()


def perform_lane_detection():
    video_cap = cv.VideoCapture(
        '/Users/siddiqui/Documents/Projects/self-drive/CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4')
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            print("cannot read frame")
            return -1
        # cv.imshow("frame", frame)
        # detect_lane(frame)


def detect_lane(image):
    # read the image
    width = image.shape[1]
    height = image.shape[0]

    # gray scale and noise reduction
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,  # source
                           (3, 3),  # kernel size
                           0)  # border type
    cv.imshow("frame", gray)
    cv.waitKey(2)
    # find the edges
    edges = cv.Canny(gray,
                     50,  # low threshold
                     150)  # high threshold

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(0, height), # left corner
                          (width/2, height/2), # apex
                          (width, height)]],  # right corner
                        dtype=np.int32)

    cv.fillPoly(mask,  # image
                vertices,  # coordinates
                ignore_mask_color)
    cv.imshow("mask", mask)
    # remove the parts of image which are not within vertices
    masked_edges = cv.bitwise_and(edges, mask)

    # find lines using hough
    detected_lines = cv.HoughLinesP(masked_edges,  # source
                                    2,  # rho --> 1 pixel
                                    # theta in radian (1 degree)
                                    1 * np.pi / 180,
                                    15,  # min voting
                                    5,  # min line length in pixels
                                    1  # max line gap in pixels
                                    )

    # creating a blank to draw lines on
    line_image = np.copy(image) * 0

    # several lines are detected
    # each line has start and end points
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image,  # source
                    (x1, y1),  # start point
                    (x2, y2),  # end point
                    (0, 0, 255),  # BGR color --> red
                    5  # line thickness
                    )

    # gray images can't draw colored features on it
    # this is a way to do it, using dstack and weighted sum
    color_edges = np.dstack((edges, edges, edges))
    line_edges = cv.addWeighted(color_edges,  # source 1
                                0.8,  # alpha --> weight of the first array elements
                                line_image,  # source 2
                                1,  # beta --> weight of the second array elements
                                0)  # gamma --> scalar added to each sum

    cv.imshow("masked edges", masked_edges)
    cv.imshow("result: ", line_edges)
    cv.waitKey()


image = cv.imread(
    '/Users/siddiqui/Documents/Projects/self-drive/CarND-LaneLines-P1/test_images/solidWhiteCurve.jpg')
detect_lane(image)
