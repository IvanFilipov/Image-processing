"""
A prove of concept body pos tracking.
"""

import cv2 as cv
import mediapipe as mp
import math

#
# cameras config - from specs
CAM_WIDTH = 720
CAM_HEIGHT = 480


def process_frame(frame):
    # convert RGB -> HSV (in order to work easier with colors)
    # hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # # equalize the histogram in L/V channel
    # hsv_frame[:, :, 2] = utils.equalize_lightness(hsv_frame)

    # # get blue pixels and remove artifacts
    # blue_mask = utils.get_clr_mask(hsv_frame, utils.Color.BLUE)
    # #
    # cv.morphologyEx(blue_mask, cv.MORPH_OPEN,\
    #     cv.getStructuringElement(cv.MORPH_RECT, (15, 15)), blue_mask)
    # cv.imshow("blue-mask", blue_mask)

    # contours, _ = cv.findContours(blue_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # if len(contours) > 0:
    #     contour = max(contours, key=cv.contourArea)
    #     (_, radius) = cv.minEnclosingCircle(contour)
    #     M = cv.moments(contour)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     # only proceed if the radius meets a minimum size
    #     if radius > 10:
    #         #cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    #         cv.circle(frame, center, 5, (0, 0, 255), -1)
    #     return center

    # cv.imshow("blue-mask", blue_mask)
    return None


def mainloop():

    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cam = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_vid = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        _, cam_frame = cam.read()

        RGB = cv.cvtColor(cam_frame, cv.COLOR_BGR2RGB)
        results = pose.process(RGB)
        mp_drawing.draw_landmarks(cam_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        

#        full_frame = cv.hconcat([l_frame, r_frame])
        cv.imshow('cam', cam_frame)
        out_vid.write(cam_frame)

        if cv.waitKey(1) == 27:
            break

    cam.release()
    out_vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    mainloop()
