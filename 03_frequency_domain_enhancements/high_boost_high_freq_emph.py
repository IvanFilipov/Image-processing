import numpy as np
import cv2   as cv
import math

"""
High boost:
enhance high frequency component while still keeping the low frequency components.

High frequency emphasis:
sharpening of an image by emphasizing the edges.
"""

CUBE_IMG = "../res/cube.jpg"
img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
M, N = img.shape

CUTOFF_FREQ = 60

# High boost filter
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
for i in range(M):
    for j in range(N):
        D = math.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
        HP = (1 - np.exp(- D * D / (2 * CUTOFF_FREQ * CUTOFF_FREQ)))
        dft_shift[i, j] = dft_shift[i, j] * ((2.2 - 1) + HP)
f_ishift = np.fft.ifftshift(dft_shift)
img_back_hbf = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_back_hbf[img_back_hbf < 0] = 0
img_back_hbf[img_back_hbf > 255] = 255


# High frequency emphasis filter
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
for i in range(M):
    for j in range(N):
        D = math.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
        HP = (1 - np.exp(- D * D / (2 * 60 * 60)))
        dft_shift[i, j] = dft_shift[i, j] * (0.5 + 2 * HP)
f_ishift = np.fft.ifftshift(dft_shift)
img_back_hfef = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_back_hfef[img_back_hfef < 0] = 0
img_back_hfef[img_back_hfef > 255] = 255

cv.imshow('HBF vs HFEF', np.concatenate((img, img_back_hbf, img_back_hfef), axis=1).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()