import numpy as np
import cv2   as cv

NOISY_IMG = "../res/noisy_cube.jpg"

mask_3 = np.ones((3, 3)) / 9
mask_5 = np.ones((5, 5)) / 25

img = cv.imread(NOISY_IMG, cv.IMREAD_COLOR)

# apply OpenCV filter2D function
output_3 = cv.filter2D(img, -1, mask_3)

# apply 5x5 mask
output_5 = cv.filter2D(img, -1, mask_5)


# show the images
cv.imshow("input", img)
cv.imshow("output 3", output_3)
cv.imshow("output 5", output_5)

cv.waitKey(0)
cv.destroyAllWindows()