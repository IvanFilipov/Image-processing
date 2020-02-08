"""Utility functions for basic image processing operations."""

import math
from enum import Enum

import numpy as np
import cv2   as cv

# HSV color constants
HSV_BLUE_LOW   = np.array([100, 0, 0], np.uint8)
HSV_BLUE_HIGH  = np.array([140, 255, 255], np.uint8)
HSV_WHITE_LOW  = np.array([0, 0, 230], np.uint8)
HSV_WHITE_HIGH = np.array([255, 255, 255], np.uint8)
#
# Morphologic closing structure elements
BLUE_CLOSE_STRUCT_EL  = np.ones((5, 10), np.uint8)
WHITE_CLOSE_STRUCT_EL = np.ones((15, 15), np.uint8)
# Morphologic opening structure element
OPENING_STRUCT_EL_BEG = 5
OPENING_STRUCT_EL_STEP = 5
OPENING_STRUCT_EL_END = 50

BGR_GREEN = (0, 255, 0)

class Color(Enum):
    """Color constants"""
    WHITE = 0
    BLUE = 1

def calc_angle(x1, y1, x2, y2):
    """Calculate angle between a given line ((x1, y1), (x2, y2))
       and the X-axis."""
    dx = (x1 - x2)
    dy = (y1 - y2)
    if dy < 0:
        alpha = math.degrees(math.atan2(dy, dx))
        if alpha < 0:
            alpha *= -1.0
    else:
        alpha = 180.0 - math.degrees(math.atan2(dy, dx))

    return alpha

def equalize_lightness(hsv_img):
    """ Equalize L/V channel in order to make
        light pixels even more lighter.
    """
    #grey = cv.split(HSV_img)[2]
    #cv.imshow("HSV-grey", grey)
    return cv.equalizeHist(hsv_img[:, :, 2])
    #grey = cv.split(HSV_img)[2]
    #cv.imshow("HSV-grey-eq", grey)

def get_clr_mask(hsv_img, clr):
    """Get a mask with specific color marked pixels in HSV image."""
    if clr == Color.BLUE:
        lower = HSV_BLUE_LOW
        upper = HSV_BLUE_HIGH
    elif clr == Color.WHITE:
        lower = HSV_WHITE_LOW
        upper = HSV_WHITE_HIGH
    else:
        raise ValueError

    return cv.inRange(hsv_img, lower, upper)

def remove_artifacts_from_mask(mask_img, clr):
    """Remove small areas of wrongly marked pixels from
       a color mask, using morphologic opening."""
    if clr == Color.BLUE:
        kernel = BLUE_CLOSE_STRUCT_EL
    elif clr == Color.WHITE:
        kernel = WHITE_CLOSE_STRUCT_EL
    else:
        raise ValueError

    cv.morphologyEx(mask_img, cv.MORPH_OPEN, kernel, mask_img)

def fill_mask(binary_mask):
    """Fill the gaps in binary mask, by applying morphologic closing,
       as iteration process, until all gaps are removed."""
    i = OPENING_STRUCT_EL_BEG
    mid_res = binary_mask
    while True:
        #print(i)
        mid_res = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, np.ones((i, i + 2), np.uint8))
        diff = cv.bitwise_xor(mid_res, binary_mask)
        # check if the step has changed the result
        if cv.countNonZero(diff) == 0 or i >= OPENING_STRUCT_EL_END:
            break
        binary_mask = mid_res
        i += OPENING_STRUCT_EL_STEP
    return binary_mask

def draw_crosshair(frame):
    """Draw a crosshair in the middle of a frame."""
    rows, cols = frame.shape[:2]
    center = (x_mid, y_mid) = (int(cols / 2), int(rows / 2))
    cv.circle(frame, center, 50, BGR_GREEN, 2)
    cv.line(frame, (x_mid, 0), (x_mid, cols - 1), BGR_GREEN, 2)
    cv.line(frame, (0, y_mid), (cols - 1, y_mid), BGR_GREEN, 2)