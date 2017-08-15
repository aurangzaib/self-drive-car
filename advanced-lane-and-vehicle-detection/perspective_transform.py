import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[345, 70],
         [340, 125],
         [465, 143],
         [466, 90]]
    )
    dst = np.float32(
        [[340, 70],
         [340, 125],
         [460, 143],
         [460, 90]]
    )
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)
    plt.imshow(img)
    plt.imshow(warped)
    plt.show()


def main__():
    im = plt.imread("/Users/siddiqui/Downloads/perspective-transform.jpg")
    warp(im)


main__()

"""
sobel operator:
sobelx --> detect edges in vertical direction
sobely --> detect edges in horizontal direction
sobely = transpose of sobelx

sobelx = cv2.sobel(image, cv2.CV_64F, 1, 0)
sobely = cv2.sobel(image, cv2.CV_64F, 0, 1)

sobel_abs = np.absolute(sobelx)
sobel_8bit= np.uint8(255*sobel_abs/np.max(sobel_abs))

create a binary threshold to select pixels based on gradient strength:
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
"""
