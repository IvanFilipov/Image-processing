"""Run the processing steps on previously taken images."""

import cv2 as cv

from common import processing

NUM_TEST_IMG = 20
SAVE_ON = True

ESC = 27
LEFT_ARROW = 83
RIGHT_ARROW = 81

IMG_BASE_NAME = "./res/img_%d.jpg"
IMG_PROC_NAME = "./res/img_proc_%d.jpg"

def empty_test():
    """Process a frame without bottles."""
    img = cv.imread("./res/img_empty.jpg", cv.IMREAD_COLOR)
    cv.imshow("img-original", img)
    processing.process_frame(img)
    cv.imshow("img-processed", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def handle_input_key(cur_index):
    key_pressed = cv.waitKey(0)
    while key_pressed not in [ESC, LEFT_ARROW, RIGHT_ARROW]:
        key_pressed = cv.waitKey(0)

    if key_pressed == ESC:
        return -1
    if key_pressed == LEFT_ARROW:
        return (cur_index + 1) % (NUM_TEST_IMG + 1)
    if key_pressed == RIGHT_ARROW:
        if cur_index == 0:
            return NUM_TEST_IMG
        else:
            return cur_index - 1

def run_tests():
    """Loop through the images and show the results."""
    i = 0
    saved = []
    while True:
        img_name = IMG_BASE_NAME % i
        img = cv.imread(img_name, cv.IMREAD_COLOR)
        cv.imshow("img-original", img)
        processing.process_frame(img)
        cv.imshow("img-processed", img)

        if SAVE_ON and i not in saved:
            cv.imwrite(IMG_PROC_NAME % i, img)
            saved.append(i)

        i = handle_input_key(i)
        if i == -1:
            break
        
    cv.destroyAllWindows()

if __name__ == "__main__":
    #empty_test()
    run_tests()