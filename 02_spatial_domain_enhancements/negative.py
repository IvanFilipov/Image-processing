import cv2 as cv

GRAY_IMG    = "../res/gray.jpg"
GRAY_EQ_IMG = "../res/gray_equalized.jpg"

gray_img = cv.imread(GRAY_IMG, cv.IMREAD_GRAYSCALE)
# apply negative transformation by using numpy matrix operations
gray_negative_img = 255 - gray_img

cv.imshow("gray", gray_img)
cv.imshow("gray negative", gray_negative_img)

cv.waitKey(0)
cv.destroyAllWindows()

# repeat the same for the equalized histogram image
gray_equalized_img = cv.imread(GRAY_EQ_IMG, cv.IMREAD_GRAYSCALE)
gray_equalized_negative_img = 255 - gray_equalized_img

cv.imshow("gray equalized", gray_equalized_img)
cv.imshow("gray equalized negative", gray_equalized_negative_img)

cv.waitKey(0)
cv.destroyAllWindows()