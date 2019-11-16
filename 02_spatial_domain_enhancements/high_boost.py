import numpy as np
import cv2 as cv

PHOTO_IMG = "../res/photo.jpg"

# define filter matrix
high_boost_4_5 = np.array([[ 0,   -1,  0],
                           [-1,  4.5, -1],
                           [ 0,   -1,  0]])
high_boost_8_5 = np.array([[-1,   -1, -1],
                           [-1,  8.5, -1],
                           [-1,   -1, -1]])

high_boost_4_9 = np.array([[ 0,   -1,  0],
                           [-1,  4.9, -1],
                           [ 0,   -1,  0]])
high_boost_8_9 = np.array([[-1,   -1, -1],
                           [-1,  8.9, -1],
                           [-1,   -1, -1]])


# read img image as grayscale image
img = cv.imread(PHOTO_IMG, cv.IMREAD_GRAYSCALE)


# apply OpenCV filter2D function
output_high_boost_4_5 = cv.filter2D(img, -1, high_boost_4_5)
output_high_boost_4_9 = cv.filter2D(img, -1, high_boost_4_9)

output_high_boost_8_5 = cv.filter2D(img, -1, high_boost_8_5)
output_high_boost_8_9 = cv.filter2D(img, -1, high_boost_8_9)


# show the images
cv.imshow('img', img)
cv.imshow('output 4.5 vs 8.5', np.concatenate((output_high_boost_4_5, output_high_boost_8_5), axis=1))
cv.imshow('output 4.5 vs 4.9', np.concatenate((output_high_boost_4_5, output_high_boost_4_9), axis=1))
cv.imshow('output 8.5 vs 8.9', np.concatenate((output_high_boost_8_5, output_high_boost_8_9), axis=1))

cv.waitKey(0)
cv.destroyAllWindows()