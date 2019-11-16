import cv2 as cv

COLOR_IMG = "../res/color.jpg"

def HSV_brightness_histogram_equalize(img):
    # convert to HSV
    HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # equalize V channel
    HSV_img[:, :, 2] = cv.equalizeHist(HSV_img[:, :, 2])
    # convert back to BGR
    return cv.cvtColor(HSV_img, cv.COLOR_HSV2BGR)

# load image
original_img = cv.imread(COLOR_IMG, cv.IMREAD_COLOR)
cv.imshow("original", original_img)
cv.imshow("processed", HSV_brightness_histogram_equalize(original_img))

cv.waitKey(0)
cv.destroyAllWindows()