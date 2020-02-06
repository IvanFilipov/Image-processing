"""Find the conturs of an object within an image,
using morphologic operations.
"""

import cv2   as cv
import numpy as np

BOOK_IMG = "../res/book.png"

img = cv.imread(BOOK_IMG, cv.IMREAD_COLOR)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("img-gray", img_gray)

_, bin_img = cv.threshold(img_gray, 80, 255, cv.THRESH_TOZERO_INV)
_, bin_img = cv.threshold(bin_img, 16, 255, cv.THRESH_TOZERO)
_, bin_img = cv.threshold(bin_img, 18, 255, cv.THRESH_BINARY)

cv.imshow("img-bin", bin_img)

kernel = np.ones((5, 8), np.uint8)
opening_5 = cv.morphologyEx(bin_img, cv.MORPH_ERODE, kernel)

mid = cv.bitwise_xor(bin_img, opening_5)
bin_img = cv.bitwise_and(bin_img, mid)

cv.imshow("opening_5", bin_img)

cv.waitKey(0)
cv.destroyAllWindows()