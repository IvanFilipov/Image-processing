"""Erosion, Dilation and Opening."""

import cv2   as cv
import numpy as np

GRAY_TEXT_IMG = "../res/gray_text.gif"

_, img = cv.VideoCapture(GRAY_TEXT_IMG).read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("img-original", img)

_, bin_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("img-bin", bin_img)


kernel = np.ones((2, 2), np.uint8)

erosion  = cv.erode(bin_img, kernel)
dilation = cv.dilate(bin_img, kernel)
# Opening is just another name of erosion followed by dilation
opening = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)

cv.imshow("erosion", erosion)
cv.imshow("dilation", dilation)
cv.imshow("opening", opening)

cv.waitKey(0)
cv.destroyAllWindows()

GRAY_CIRCUIT_IMG = "../res/circuit.gif"
_, img = cv.VideoCapture(GRAY_CIRCUIT_IMG).read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("img-original", img)

kernel = np.ones((5, 5), np.uint8)
opening_5 = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

kernel = np.ones((9, 9), np.uint8)
opening_9 = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

cv.imshow("opening-5", opening_5)
cv.imshow("opening-9", opening_9)

cv.waitKey(0)
cv.destroyAllWindows()