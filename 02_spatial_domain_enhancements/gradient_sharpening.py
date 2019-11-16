import numpy as np
import cv2   as cv

PHOTO_IMG = "../res/photo.jpg"

# define filter matrix
sobel_h = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
sobel_v = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])

roberts_m = np.array([[-1,  0],
                      [ 0,  1]])
roberts_s = np.array([[ 0, -1],
                      [ 1,  0]])


# read input image as grayscale image
img = cv.imread(PHOTO_IMG, cv.IMREAD_GRAYSCALE)

# apply OpenCV filter2D function
output_sobel_h = cv.filter2D(img, -1, sobel_h)
output_sobel_v = cv.filter2D(img, -1, sobel_v)

output_roberts_m = cv.filter2D(img, -1, roberts_m)
output_roberts_s = cv.filter2D(img, -1, roberts_s)


# show the images
cv.imshow('input', img)
cv.imshow('sobel', np.concatenate((output_sobel_h, output_sobel_v), axis=1))
cv.imshow('roberts', np.concatenate((output_roberts_m, output_roberts_s), axis=1))

cv.waitKey(0)
cv.destroyAllWindows()