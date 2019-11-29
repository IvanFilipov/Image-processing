import numpy as np
import cv2   as cv
import math

"""Low pass:  “Пропуска“ само ниските честоти.
              Намалява детайлите, размива контурите.
   High pass: „Пропуска“ само високите честоти.
              Подчертава контурите.
"""

CUBE_IMG = "../res/cube.jpg"
img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
M, N = img.shape

CUTOFF_FREQ = 60

def ideal_pass_filter(img_dft_shift, is_low):
    ipf_shift = np.copy(img_dft_shift)
    for i in range(M):
        for j in range(N):
            calc = math.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
            if (is_low and calc > CUTOFF_FREQ) or\
               (not is_low and calc < CUTOFF_FREQ):
                ipf_shift[i, j] = 0
        
    f_ishift = np.fft.ifftshift(ipf_shift)
    img_back_ipf = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    img_back_ipf[img_back_ipf < 0] = 0
    img_back_ipf[img_back_ipf > 255] = 255

    return img_back_ipf

img_back_ilpf = ideal_pass_filter(dft_shift, is_low=True)
img_back_ihpf = ideal_pass_filter(dft_shift, is_low=False)

cv.imshow('ILPF vs IHPF', np.concatenate((img, img_back_ilpf, img_back_ihpf), axis=1).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()