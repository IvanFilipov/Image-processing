import numpy as np
import cv2   as cv

"""намалява средната стойност на интензитета до „нула“"""

CUBE_IMG = "../res/cube.jpg"
img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


# Notch filter
M, N = img.shape

dft_shift[M // 2, N // 2] = 0

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_back[img_back < 0] = 0
img_back[img_back > 255] = 255

cv.imshow('notch filter', np.concatenate((img, img_back), axis=1).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()