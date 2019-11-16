import cv2 as cv

COLOR_IMG = "../res/color.jpg"

def RGB_histogram_equalize(img):
    b_ch, g_ch, r_ch = cv.split(img) # get all chanels
    red_hist   = cv.equalizeHist(r_ch)
    green_hist = cv.equalizeHist(g_ch)
    blue_hist  = cv.equalizeHist(b_ch)
    return cv.merge((blue_hist, green_hist, red_hist))

# load image
original_img = cv.imread(COLOR_IMG, cv.IMREAD_COLOR)
cv.imshow("original", original_img)
cv.imshow("processed", RGB_histogram_equalize(original_img))

cv.waitKey(0)
cv.destroyAllWindows()