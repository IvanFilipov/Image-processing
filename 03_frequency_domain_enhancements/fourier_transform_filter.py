import numpy as np
import cv2   as cv
from matplotlib import pyplot as plt

NOISY_IMG = "../res/noisy_cube.jpg"

img = cv.imread(NOISY_IMG, cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

rms = 100
cms = 100

mask = np.zeros((rows, cols, 2), np.uint8)

# inner
mask[crow-rms:crow+rms, ccol-cms:ccol+cms] = 1
# on sides
mask[0:rms, :] = 1
mask[rows-rms:rows, :] = 1
mask[:, 0:cms] = 1
mask[:, cols-cms:cols] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
mask_spectrum = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(mask_spectrum, cmap='gray')
plt.title('Masked magnitude')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(img_back, cmap='gray')
plt.title('Output')
plt.xticks([])
plt.yticks([])
plt.show()