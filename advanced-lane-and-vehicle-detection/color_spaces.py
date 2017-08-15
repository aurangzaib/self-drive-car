import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

thresh = (170, 255)
img = mpimg.imread("/Users/siddiqui/Downloads/test4.jpg")

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray_binary = np.zeros_like(gray)
gray_binary[(gray >= 20) & (gray <= 80)] = 1

# sobelx gradient threshold
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

sobelx_abs = np.absolute(sobelx)
sobelx_8bit = np.uint8(255 * sobelx_abs / np.max(sobelx_abs))
sobelx_binary = np.zeros_like(sobelx_8bit)
sobelx_binary[(sobelx_8bit > 20) & (sobelx_8bit <= 200)] = 1

# RGB color space
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
R_binary = np.zeros_like(R)
R_binary[(R >= thresh[0]) & (R <= thresh[1])] = 1

# HLS color space
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
H, L, S = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
S_binary = np.zeros_like(S)
S_binary[(S >= 120) & (S <= 255)] = 1
H_binary = np.zeros_like(H)
H_binary[(H >= 15) & (H <= 100)] = 1

# combine results of sobel and S channel
sobel_S_resultant = np.zeros_like(sobelx_binary)
sobel_S_resultant[((sobelx_binary == 1) | (S_binary == 1)) & (R_binary == 1)] = 1
sobel_S_stack = np.dstack((np.zeros_like(sobelx_binary), sobelx_binary, S_binary))

# visualization
f, ax_array = plt.subplots(4, 2, figsize=(7, 7))
f.tight_layout()
ax_array[0, 0].imshow(R, cmap="gray"), ax_array[0, 0].set_title("R")
ax_array[0, 1].imshow(R_binary, cmap="gray"), ax_array[0, 1].set_title("R binary")
ax_array[1, 0].imshow(S, cmap="gray"), ax_array[1, 0].set_title("S")
ax_array[1, 1].imshow(S_binary, cmap="gray"), ax_array[1, 1].set_title("S_binary")
ax_array[2, 0].imshow(sobelx, cmap="gray"), ax_array[2, 0].set_title("sobelx")
ax_array[2, 1].imshow(sobelx_binary, cmap="gray"), ax_array[2, 1].set_title("sobelx_binary")
ax_array[3, 0].imshow(sobel_S_resultant, cmap="gray"), ax_array[3, 0].set_title("sobelx and S")
ax_array[3, 1].imshow(sobel_S_stack), ax_array[3, 1].set_title("sobelx and S stack")

# open the plot window
plt.show()
