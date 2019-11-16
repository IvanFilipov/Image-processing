import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

GRAY_IMG    = "../res/gray.jpg"
GRAY_EQ_IMG = "../res/gray_equalized.jpg"

# load image to numpy array by using OpenCV
gray_img = cv.imread(GRAY_IMG, cv.IMREAD_GRAYSCALE)
gray_equalized_img = cv.imread(GRAY_EQ_IMG, cv.IMREAD_GRAYSCALE)


fig, axes = plt.subplots(nrows=1, ncols=2)
ax0, ax1  = axes.flatten()

# compute and display histogram
ax0.hist(gray_img, bins=range(257), histtype='bar')
ax0.set_title("gray_histogram")
ax0.set_xlim((0, 255))
ax0.set_ylim((0, 3000))

# apply contrast stretching by using numpy matrix operations
# Note: do not forget to cast result to 8-bit unsigned int
a = 0   # min intensity
b = 255 # max intensity
c = 115 # lower strech bound
d = 190 # upper strech bound
gray_constrast_streched_img = (gray_img.astype(np.double) - c) * (b - a) / (d - c) + a
gray_constrast_streched_img[gray_constrast_streched_img < a] = a
gray_constrast_streched_img[gray_constrast_streched_img > b] = b
gray_constrast_streched_img = gray_constrast_streched_img.astype(np.uint8)

# compute and display streched histogram
ax1.hist(gray_constrast_streched_img, bins=range(257), histtype='bar')
ax1.set_title("streched histogram")
ax1.set_xlim((0, 255))
ax1.set_ylim((0, 3000))

fig.tight_layout()
plt.show()

# show the images
cv.imshow("input", gray_img)
cv.imshow("input equalized", gray_equalized_img)
cv.imshow("gray constrast streched", gray_constrast_streched_img)

cv.waitKey(0)
cv.destroyAllWindows()