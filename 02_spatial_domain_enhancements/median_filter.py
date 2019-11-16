import numpy as np
import matplotlib.pyplot as plt
import cv2

import numpy as np
import cv2   as cv

MEDIAN_IMG = "../res/median.jpeg"

img = cv.imread(MEDIAN_IMG, cv.IMREAD_COLOR)

gauss  = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)

images = np.concatenate((img, gauss, median), axis=1)

# show outputs
cv.imshow('output', images)
cv.waitKey(0)
cv.destroyAllWindows()