import cv2   as cv
import numpy as np

GRAY_EQ_IMG = "../res/gray_equalized.jpg"

gray_equalized_img = cv.imread(GRAY_EQ_IMG, cv.IMREAD_GRAYSCALE)
# apply gamma correction by using numpy matrix operations
# Note: do not forget to cast result to 8-bit unsigned int
gamma_2   = (255.0 * np.power(gray_equalized_img / 255.0, 2)).astype(np.uint8)
gamma_0_5 = (255.0 * np.power(gray_equalized_img / 255.0, 0.5)).astype(np.uint8)

cv.imshow("input", gray_equalized_img)
cv.imshow("gamma correction 2", gamma_2)
cv.imshow("gamma correction 0.5", gamma_0_5)

cv.waitKey(0)
cv.destroyAllWindows()