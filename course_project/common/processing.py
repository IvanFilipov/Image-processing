"""Processing steps used for bottle detection and marking."""

import numpy as np
import cv2   as cv

from common import utils

BLUE_PIXELS_THRESHOLD = 12000

BGR_RED = (0, 0, 255)
BGR_WHITE = (255, 255, 255)
BGR_GREEN = (0, 255, 0)
BGR_ORANGE = (0, 165, 255)

TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX

def draw_obj_angle(frame, contour, object_id, angle):
    """Put the object id and angle on the frame."""
    M = cv.moments(contour)
    center = (int(M["m10"] / M["m00"]) + 5, int(M["m01"] / M["m00"]))
    text = "O[%d], angle: %0.2f" % (object_id, angle)
    cv.putText(frame, text, center, TEXT_FONT, 0.7, BGR_ORANGE, 2, cv.LINE_AA)

def draw_center(frame, contour):
    M = cv.moments(contour)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cv.circle(frame, center, 5, BGR_RED, -1)


def draw_features(frame, objects_mask):
    """By a mask of known objects - draw their features -
       contour, angle..."""
    # extract contours
    contours, _ = cv.findContours(objects_mask, cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
    # draw them
    #TODO: don't draw contours if object area is two small.
    cv.drawContours(frame, contours, -1, BGR_RED, 5)
    # create lines in objects' directions
    i = 0
    for contour in contours:
        # create line through whole frame
        rows, cols = frame.shape[:2]
        [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        # line end points
        x1, y1 = (cols- 1, righty)
        x2, y2 = (0, lefty)
        # trim it to be only inside the object
        line_mask = np.zeros((rows, cols), dtype=np.uint8) # create black mask
        line_mask = cv.line(line_mask, (x1, y1), (x2, y2), BGR_WHITE, 2) # while line
        line_mask = cv.bitwise_and(line_mask, objects_mask) # combine it with object's mask
        frame[line_mask == 255] = BGR_GREEN # draw it on the frame
        #
        draw_obj_angle(frame, contour, i, utils.calc_angle(x1, y1, x2, y2))
        i += 1
    
    if len(contours) > 0:
        draw_center(frame, contours[0])


def process_frame(frame):
    """Apply all operations on a single frame."""
    # convert RGB -> HSV (in order to work easier with colors)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # equalize the histogram in L/V channel
    hsv_frame[:, :, 2] = utils.equalize_lightness(hsv_frame)

    # get blue pixels and remove artifacts
    blue_mask = utils.get_clr_mask(hsv_frame, utils.Color.BLUE)
    cv.imshow("blue-mask", blue_mask)
    utils.remove_artifacts_from_mask(blue_mask, utils.Color.BLUE)
    #cv.imshow("blue-mask", blue_mask)
    if cv.countNonZero(blue_mask) < 12000:
        return

    # get white(most lit pixels)
    white_mask = utils.get_clr_mask(hsv_frame, utils.Color.WHITE)
    utils.remove_artifacts_from_mask(white_mask, utils.Color.WHITE)

    # combine the two masks
    objects_mask = cv.bitwise_or(white_mask, blue_mask)
    # cv.imshow("objects-mask", objects_mask)
    # remove the gaps
    objects_mask = utils.fill_mask(objects_mask)
    #cv.imshow("objects-mask", objects_mask)

    draw_features(frame, objects_mask)
