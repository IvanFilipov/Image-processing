import numpy as np
import cv2   as cv

PHOTO_IMG = "../res/photo.jpg"

# define filter matrix
laplace_4 = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]])
laplace_8 = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]])

composite_4 = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
composite_8 = np.array([[-1, -1,  -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])


# read input image as grayscale image
img = cv.imread(PHOTO_IMG, cv.IMREAD_GRAYSCALE)


# apply OpenCV filter2D function
output_laplace_4 = cv.filter2D(img, -1, laplace_4)
output_laplace_8 = cv.filter2D(img, -1, laplace_8)

output_composite_4 = cv.filter2D(img, -1, composite_4)
output_composite_8 = cv.filter2D(img, -1, composite_8)


# show the images
cv.imshow('input', img)
cv.imshow('laplace',  np.concatenate((output_laplace_4, output_laplace_8), axis=1))
cv.imshow('composite',  np.concatenate((output_composite_4, output_composite_8), axis=1))

cv.waitKey(0)
cv.destroyAllWindows()