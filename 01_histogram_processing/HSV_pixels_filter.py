import numpy as np
import cv2   as cv

RGB_IMG  = "../res/rgb.png"

def solve_task7():
    img = cv.imread(RGB_IMG, cv.IMREAD_COLOR)
    cv.imshow("original", img)
    rgb_mat_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # filter_ = np.copy(rgb_mat_hsv)
    # each color > 50 to become zero
    red = np.uint8([[[0, 0, 255]]])
    hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
    rh = hsv_red[0][0][0]

    upper_red = np.array([rh + 10, 255, 255])
    lower_red = np.array([max(rh - 10, 0), 100, 100])

    # Threshold the HSV image to get only red colors
    mask = cv.inRange(rgb_mat_hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(img, img, mask=mask)

    #filter[filter[:,:,0] > 320] = 0
    #filter = cv.cvtColor(filter, cv.COLOR_HSV2BGR)
    cv.imshow("mask", mask)
    cv.imshow("filtered", res)


if __name__ == '__main__':
    solve_task7()
    cv.waitKey(0)
    cv.destroyAllWindows()
