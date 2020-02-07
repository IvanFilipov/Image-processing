"""Run the processing steps on previously taken images."""

import cv2 as cv

from common import processing

NUM_TEST_IMG = 20

def empty_test():
    """Process a frame without bottles."""
    img = cv.imread("./res/img_empty.jpg", cv.IMREAD_COLOR)
    cv.imshow("img-original", img)
    processing.process_frame(img)
    cv.imshow("img-processed", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def run_tests():
    """Loop through the images and show the results."""
    for i in range(0, 1):#NUM_TEST_IMG + 1):
        img_name = "./res/img_%d.jpg" % i
        img = cv.imread(img_name, cv.IMREAD_COLOR)
        cv.imshow("img-original", img)
        processing.process_frame(img)
        cv.imshow("img-processed", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    #empty_test()
    run_tests()