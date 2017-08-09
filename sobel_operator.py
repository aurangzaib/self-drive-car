import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def sobel_absolute_threshold(img, orient, sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    dx, dy = (1, 0) if orient is 'x' else (0, 1)
    sobel = cv2.Sobel(img_gray, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
    # 3) Take the absolute value of the directional gradient
    sobel_abs = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_8bit = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    #    is > thresh_min and < thresh_max
    sobel_binary = np.zeros_like(sobel_8bit)
    sobel_binary[(sobel_8bit >= thresh_min) & (sobel_8bit <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


def sobel_magnitude_threshold(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x and y
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the magnitude of gradient
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_8bit = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    #    is >= thresh_min and <= thresh_max
    sobel_binary = np.zeros_like(sobel_8bit)
    sobel_binary[(sobel_8bit >= thresh_min) & (sobel_8bit <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


def sobel_orientation_threshold(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    # Apply the following steps to img
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x and y
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute sobel
    sobelx_abs, sobely_abs = np.absolute(sobelx), np.absolute(sobely)
    # 4) Direction of the Gradient
    orientation = np.arctan2(sobely_abs, sobelx_abs)
    # 5) Create a mask of 1's where the gradient direction
    #    is >= thresh_min and <= thresh_max
    sobel_binary = np.zeros_like(orientation)
    sobel_binary[(orientation >= thresh_min) & (orientation <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


__ = mpimg.imread("/Users/siddiqui/Downloads/signs_vehicles_xygrad.jpg")
grad_x = sobel_absolute_threshold(__, 'x', 9, 20, 80)
grad_y = sobel_absolute_threshold(__, 'y', 9, 20, 80)
grad_magnitude = sobel_magnitude_threshold(__, 9, 20, 80)
grad_orientation = sobel_orientation_threshold(__, 15, 0.7, 1.3)
grad_resultant = np.zeros_like(grad_orientation)
grad_resultant[((grad_x == 1) & (grad_y == 1)) | ((grad_magnitude == 1) & (grad_orientation == 1))] = 1
f, ax_array = plt.subplots(2, 3, figsize=(7, 7))
f.tight_layout()

ax_array[0, 0].imshow(__)
ax_array[0, 0].set_title("original")

ax_array[0, 1].imshow(grad_x, cmap="gray")
ax_array[0, 1].set_title("sobel x")

ax_array[0, 2].imshow(grad_y, cmap="gray")
ax_array[0, 2].set_title("sobel y")

ax_array[1, 0].imshow(grad_magnitude, cmap="gray")
ax_array[1, 0].set_title("gradient magnitude")

ax_array[1, 1].imshow(grad_orientation, cmap="gray")
ax_array[1, 1].set_title("gradient orientation")

ax_array[1, 2].imshow(grad_resultant, cmap="gray")
ax_array[1, 2].set_title("gradient resultant")

plt.show()
