import numpy as np
import cv2   as cv
import math

"""“Пропуска“ само ниските честоти.
Намалява детайлите, размива контурите.
"""

CUBE_IMG = "../res/cube.jpg"
img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
M, N = img.shape


# Notch filter
notch_shift = np.copy(dft_shift)
notch_shift[M // 2, N // 2] = 0
f_ishift = np.fft.ifftshift(notch_shift)
img_back_notch = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_back_notch[img_back_notch < 0] = 0
img_back_notch[img_back_notch > 255] = 255

# Ideal low pass filter
CUTOFF_FREQ = 50
ilpf_shift = np.copy(dft_shift)
for i in range(M):
    for j in range(N):
        if math.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2) > CUTOFF_FREQ:
            ilpf_shift[i, j] = 0
f_ishift = np.fft.ifftshift(ilpf_shift)
img_back_ilpf = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_back_ilpf[img_back_ilpf < 0] = 0
img_back_ilpf[img_back_ilpf > 255] = 255

cv.imshow('notch vs ILPF', np.concatenate((img, img_back_notch, img_back_ilpf), axis=1).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()