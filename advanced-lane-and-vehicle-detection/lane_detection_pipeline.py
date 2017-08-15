import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


def pipeline(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # sobelx
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    sobelx_abs = np.absolute(sobelx)
    sobelx_8bit = np.uint8(255 * sobelx_abs / np.max(sobelx))
    sobelx_binary = np.zeros_like(sobelx_8bit)
    sobelx_binary[(sobelx_8bit > 10) & (sobelx_8bit <= 100)] = 1
    # HSV channel
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2]
    S_binary = np.zeros_like(S)
    S_binary[(S > 170) & (S <= 255)] = 1
    # combine S and sobelx
    resultant = np.zeros_like(sobelx_binary)
    resultant[(sobelx_binary == 1) | (S_binary == 1)] = 1
    # visualization
    f, ax_array = plt.subplots(3, 1, figsize=(7, 7))
    f.tight_layout()
    ax_array[0].set_title("sobelx"), ax_array[0].imshow(sobelx_binary, cmap="gray")
    ax_array[1].set_title("S channel"), ax_array[1].imshow(S_binary, cmap="gray")
    ax_array[2].set_title("Resultant"), ax_array[2].imshow(resultant, cmap="gray")
    plt.show()


__ = mpimg.imread("/Users/siddiqui/Downloads/test6.jpg")
pipeline(__)
