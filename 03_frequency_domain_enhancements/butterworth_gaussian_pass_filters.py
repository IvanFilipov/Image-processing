import numpy as np
import cv2   as cv
import math

"""
Low/High pass:
Замазва/Подчертава контурите.
"""

CUBE_IMG = "../res/cube.jpg"
img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
M, N = img.shape

CUTOFF_FREQ = 60

def ideal_low_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    if math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2) > CUTOFF_FREQ:
            lpf_shift[i_ind, j_ind] = 0

def ideal_high_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    if math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2) < CUTOFF_FREQ:
            lpf_shift[i_ind, j_ind] = 0

def butterworth_low_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    D = math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2)
    lpf_shift[i_ind, j_ind] = lpf_shift[i_ind, j_ind] / (1 + (D / CUTOFF_FREQ) ** (2 * 2))

def butterworth_high_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    D = math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2)
    lpf_shift[i_ind, j_ind] = lpf_shift[i_ind, j_ind] / (1 + (D / CUTOFF_FREQ) ** (2 * 2))

    if D > 0:
        lpf_shift[i_ind, j_ind] = lpf_shift[i_ind, j_ind] / (1 + (CUTOFF_FREQ / D) ** (2 * 2))
    else:
        lpf_shift[i_ind, j_ind] = 0

def gaussian_low_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    D = math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2)
    lpf_shift[i_ind, j_ind] = lpf_shift[i_ind, j_ind] * np.exp(- D * D / (2 * CUTOFF_FREQ * CUTOFF_FREQ))

def gaussian_high_pass_calc_fn(i_ind, j_ind, N_size, M_size, lpf_shift):
    D = math.sqrt((i_ind - M_size / 2) ** 2 + (j_ind - N_size / 2) ** 2)
    lpf_shift[i_ind, j_ind] = lpf_shift[i_ind, j_ind] * (1 - np.exp(- D * D / (2 * CUTOFF_FREQ * CUTOFF_FREQ)))

def pass_filter(img_dft_shift, calc_fn):
    pf_shift = np.copy(img_dft_shift)
    for i in range(M):
        for j in range(N):
            calc_fn(i, j, N, M, pf_shift)
        
    f_ishift = np.fft.ifftshift(pf_shift)
    img_back_fp = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    img_back_fp[img_back_fp < 0] = 0
    img_back_fp[img_back_fp > 255] = 255

    return img_back_fp

img_back_ilpf = pass_filter(dft_shift, ideal_low_pass_calc_fn)
img_back_blpf = pass_filter(dft_shift, butterworth_low_pass_calc_fn)
img_back_glpf = pass_filter(dft_shift, gaussian_low_pass_calc_fn)

img_back_ihpf = pass_filter(dft_shift, ideal_high_pass_calc_fn)
img_back_bhpf = pass_filter(dft_shift, butterworth_high_pass_calc_fn)
img_back_ghpf = pass_filter(dft_shift, gaussian_high_pass_calc_fn)

cv.imshow('ILPF vs Butterworth LPF vs Gaussian LPF', 
    np.concatenate((img, img_back_ilpf, img_back_blpf, img_back_glpf),
    axis=1).astype(np.uint8))

cv.imshow('IHPF vs Butterworth HPF vs Gaussian HPF', 
    np.concatenate((img, img_back_ihpf, img_back_bhpf, img_back_ghpf),
    axis=1).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()