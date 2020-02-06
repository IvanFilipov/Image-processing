"""Image binarization with tresholds,by Otsu's method and
   Floyd-Steinberg error diffusion algorithm.
"""

import numpy             as np
import cv2               as cv
import matplotlib.pyplot as plt

GRAY_IMG = "../res/photo.gif"

_, img = cv.VideoCapture(GRAY_IMG).read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("img-original", img)


histogram = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(histogram, color='b')
plt.xlim([0, 256])
plt.show()

def binarization_with_threshold(image, treshold):
    # with openCV:
    # ret, threshed_32 = cv.threshold(img, 32, 255, cv.THRESH_BINARY)
    # cv.imshow("bin-32", threshed_32)
    bin_img = np.copy(image)
    bin_img[bin_img < treshold] = 0   # black
    bin_img[bin_img > treshold] = 255 # white
    cv.imshow("bin-%d" % treshold, bin_img)

def otsu_binarization(image, hist):
    # Otsu's thresholding
    hist = np.array([int(h[0]) for h in hist])
    # openCV's
    ret2, otsu_thr = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("otsu-opencv-%d" % int(ret2), otsu_thr)

    cur_max = 0
    cur_max_t = 0
    for t in range(0, 256):
        obj_pixels = np.sum(hist[t + 1:])
        bgr_pixles = np.sum(hist[:t + 1])
        if obj_pixels == 0 or bgr_pixles == 0:
            continue
        obj_mean = np.sum(hist[t + 1:] * np.arange(t + 1, 256)) / obj_pixels
        bgr_mean = np.sum(hist[:t + 1] * np.arange(0, t + 1)) / bgr_pixles
        cur_otsu = obj_pixels * bgr_pixles * ((bgr_mean - obj_mean) ** 2)
        if cur_otsu > cur_max:
            cur_max = cur_otsu
            cur_max_t = t       

    binarization_with_threshold(image, cur_max_t)
    return cur_max_t


def floyd_steinberg_dithering(image, treshold):
    M, N = image.shape
    for y in range(0, N - 1):
        for x in range(0, M - 1):
            if image[x][y] <= treshold:
                pixel = 0
            else:
                pixel = 255
            err = min(image[x][y], 255 - image[x][y])
            image[x][y] = pixel
            image[x + 1][y] += int(7 * err / 16)
            image[x - 1][y + 1] += int(3 * err / 16)
            image[x][y + 1] += int(5 * err / 16)
            image[x + 1][y + 1] += int(1 * err / 16)

binarization_with_threshold(img, 32)
binarization_with_threshold(img, 64)
binarization_with_threshold(img, 128)

best_threshold = otsu_binarization(img, histogram)

floyd_steinberg_dithering(img, best_threshold)
cv.imshow("dithering", img)

cv.waitKey(0)
cv.destroyAllWindows()
