import numpy as np
import cv2   as cv

PHOTO_IMG = "../res/photo.jpg"

# define filter matrix
gaus_3 = (1.0 / 16.0) * np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])

gaus_5 = (1.0 / 243.0) * np.array([[1, 4, 7, 4, 1],
                                  [4, 16, 26, 16, 4],
                                  [7, 26, 41, 26, 7],
                                  [4, 16, 26, 16, 4],
                                  [1, 4, 7, 4, 1]])

img = cv.imread(PHOTO_IMG, cv.IMREAD_COLOR)

output_3 = cv.filter2D(img, -1, gaus_3)
output_5 = cv.filter2D(img, -1, gaus_5)
output_3_open_cv = cv.GaussianBlur(img, (3, 3), 0)

cv.imshow('input', img)
cv.imshow('output 3', output_3)
cv.imshow('output 5', output_5)
cv.imshow('output 3 open CV', output_3_open_cv)

cv.waitKey(0)
cv.destroyAllWindows()