"""
A prove of concept script for distance measurement with stereo cameras.

Inspirated by: https://www.youtube.com/watch?v=sW4CVI51jDY

"""

import cv2 as cv
import math

from common import utils

# app config
CALIBRATE_ONLY = False
REC_ON = True
#
# cameras config - from specs
CAM_WIDTH = 640 # 720
CAM_HEIGHT = 480
CAM_W_ANGLE = 45 # 60
CAM_H_ANGLE = 30
# from the set up
BETWEEN_CAM_DISTANCE = 18.5
#

class DistanceProcessor:

    def __init__(self, pixel_width, pixel_height, angle_width, angle_height):
        self._width = pixel_width
        self._height = pixel_height
        self._x_angle = angle_width
        self._y_angle = angle_height

        self._mid_x = int(self._width / 2)
        self._mid_y = int(self._height / 2)

        # distance from cam to frame
        self._x_dist = self._mid_x / math.tan(math.radians(self._x_angle / 2)) 
        self._y_dist = self._mid_y / math.tan(math.radians(self._y_angle / 2))

    def calc_angles_from_center(self, x, y):
        # x, y are coordinates of a point, with top-left (0, 0)
        x = x - self._mid_x
        y = self._mid_y - y

        x_tan = x / self._x_dist
        y_tan = y / self._y_dist

        x_rad = math.atan(x_tan)
        y_rad = math.atan(y_tan)

        return (math.degrees(x_rad), math.degrees(y_rad))

    def _triangulate(self, left_x_angle, right_x_angle):
        # convert to radians
        left_x_angle = math.radians(left_x_angle)
        right_x_angle = math.radians(right_x_angle)
        # fix angle orientation (from center frame)
        left_x_angle = (math.pi / 2) - left_x_angle
        right_x_angle = (math.pi / 2) + right_x_angle

        # use the idea that BETWEEN_CAM_DISTANCE = ( d / left_tan + d / right_tan )
        dist_z = BETWEEN_CAM_DISTANCE / ((1 / math.tan(left_x_angle)) + (1 / math.tan(right_x_angle)))
        # get X measure from left-camera-center
        dist_x = dist_z / math.tan(left_x_angle)

        return dist_x, dist_z

    def find_distance(self, left_x_angle, left_y_angle, right_x_angle):

        dist_x, dist_z = self._triangulate(left_x_angle, right_x_angle)

        dist_y = math.tan(math.radians(left_y_angle)) *\
            math.sqrt(dist_x * dist_x + dist_z * dist_z)
        # complate 3D distance (we assume that dist_x, dist_y, dist_z
        #  are radius vectors from left-cam-center)
        return math.sqrt(sum([x * x for x in [dist_x, dist_y, dist_z]]))       

def process_frame(frame):
    # convert RGB -> HSV (in order to work easier with colors)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # equalize the histogram in L/V channel
    hsv_frame[:, :, 2] = utils.equalize_lightness(hsv_frame)

    # get blue pixels and remove artifacts
    blue_mask = utils.get_clr_mask(hsv_frame, utils.Color.BLUE)
    #
    cv.morphologyEx(blue_mask, cv.MORPH_OPEN,\
        cv.getStructuringElement(cv.MORPH_RECT, (15, 15)), blue_mask)
    cv.imshow("blue-mask", blue_mask)

    contours, _ = cv.findContours(blue_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = max(contours, key=cv.contourArea)
        (_, radius) = cv.minEnclosingCircle(contour)
        M = cv.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            #cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)
        return center
    
    #cv.imshow("blue-mask", blue_mask)
    return None

def compute_distance(dist_pr, c_left, c_right):
    l_target_x, l_target_y = c_left
    r_target_x, r_target_y = c_right

    l_x_angle, l_y_angle = dist_pr.\
        calc_angles_from_center(l_target_x, l_target_y)

    r_x_angle, _ = dist_pr.\
        calc_angles_from_center(r_target_x, r_target_y)

    return dist_pr.find_distance(l_x_angle, l_y_angle, r_x_angle)

def draw_distance_text(l_frame, d_text):
    """Add text to the frame."""
    cv.putText(l_frame, "Distance: " + d_text, (20, 20),\
               cv.FONT_HERSHEY_SIMPLEX, 0.8,\
               (0, 0, 255), 1, cv.LINE_AA)


def distance_processing(l_frame, r_frame, dist_pr):
    """Compute distance, if possible."""
    c_left = process_frame(l_frame)
    c_right = process_frame(r_frame)

    if c_left is None or c_right is None:
        draw_distance_text(l_frame, "-")
    else:
        distance = compute_distance(dist_pr, c_left, c_right)
        draw_distance_text(l_frame, "%0.2f" % distance)

def mainloop():
    """Get frame stream from both cameras, use it
       for calibration or distance measurement."""
    left_cam  = cv.VideoCapture(2)
    right_cam = cv.VideoCapture(4)

    dist_pr = DistanceProcessor(CAM_WIDTH, CAM_HEIGHT, CAM_W_ANGLE, CAM_H_ANGLE)

    if REC_ON:
        out_vid = cv.VideoWriter(
            'output.mp4', 0x7634706d, 20.0,
            (CAM_WIDTH * 2, CAM_HEIGHT)
        )

    while True:
        _, l_frame = left_cam.read()
        _, r_frame = right_cam.read()

        utils.draw_crosshair(l_frame)
        utils.draw_crosshair(r_frame)

        if not CALIBRATE_ONLY:
            distance_processing(l_frame, r_frame, dist_pr)

        full_frame = cv.hconcat([l_frame, r_frame])
        cv.imshow('full vision', full_frame)
        
        if REC_ON:
            out_vid.write(full_frame)

        if cv.waitKey(1) == 27:
            break

    left_cam.release()
    right_cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    mainloop()
