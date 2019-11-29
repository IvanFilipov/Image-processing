"""Improve a gray scale image contrast by histogram equalization."""

import cv2 as cv

GRAY_IMG = "../res/gray.jpg"

img = cv.imread(GRAY_IMG, cv.IMREAD_GRAYSCALE)
res = cv.equalizeHist(img)
cv.imshow("img-original",  img)
cv.imshow("img-processed", res)

cv.waitKey(0)
cv.destroyAllWindows()