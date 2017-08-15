import cv2 as cv
import numpy as np

"""
find the corners of the image using cv.findChessboardCorners
find camera matrix and distortion coef. using cv.calibrateCamera with corners and pattern size as arguments
undistort the image using cv.undistort with camera matrix and distortion coef. as arguments
"""


def get_undistorted_image(nx, ny, image, obj_pts, obj_pt, img_pts):
    # find the corners
    found, corners = cv.findChessboardCorners(image=image, patternSize=(nx, ny))
    print("# of corners: {}".format(len(corners)))
    if found is True:
        img_pts.append(corners)
        obj_pts.append(obj_pt)
        # draw the found corner points in the image
        draw_pts = np.copy(image)
        cv.drawChessboardCorners(image=draw_pts, patternSize=(nx, ny), corners=corners, patternWasFound=found)
        # find camera matrix and distortion coef.
        ret, camera_matrix, dist_coef, rot_vector, trans_vector = cv.calibrateCamera(objectPoints=obj_pts,
                                                                                     imagePoints=img_pts,
                                                                                     imageSize=image.shape[0:2],
                                                                                     cameraMatrix=None,
                                                                                     distCoeffs=None)
        # undistort the image
        undistorted_image = cv.undistort(src=image,
                                         cameraMatrix=camera_matrix,
                                         distCoeffs=dist_coef,
                                         dst=None,
                                         newCameraMatrix=camera_matrix)
        # grayscale
        gray_image = cv.cvtColor(undistorted_image, cv.COLOR_BGR2GRAY, dstCn=3)
        # gray_image = np.zeros_like(draw_pts)
        # gray_image[:, :, 0], gray_image[:, :, 1], gray_image[:, :, 2] = __, __, __
        # perspective transform
        offset = 100
        img_size = (gray_image.shape[1], gray_image.shape[0])
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        dst = np.float32([
            [offset, offset],
            [img_size[0] - offset, offset],
            [img_size[0] - offset, img_size[1] - offset],
            [offset, img_size[1] - offset]
        ])
        # perspective matrix
        M = cv.getPerspectiveTransform(src, dst)
        # Minv = cv.getPerspectiveTransform(dst, src) --> inverse perspective
        warped = cv.warpPerspective(gray_image, M, img_size, flags=cv.INTER_LINEAR)
        cv.imshow("original", image)
        cv.imshow("undistorted", gray_image)
        cv.imshow("undistorted and transformed", warped)
        cv.waitKey()


# print(gray.shape[:-1])    --> give last
# print(gray.shape[::-1])   --> give all except last

image = cv.imread("/Users/siddiqui/Downloads/chess-iphone.jpg")
nx, ny, channels = 9, 6, 3

# img_pts --> 2D in image
# obj_pts --> 3D in real world
img_pts, obj_pts = [], []

# to create a matrix of 4x5 --> np.mgrid[0:4, 0:5]
obj_pt = np.zeros(shape=(nx * ny, channels), dtype=np.float32)
obj_pt[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

get_undistorted_image(nx, ny, image, obj_pts, obj_pt, img_pts)
