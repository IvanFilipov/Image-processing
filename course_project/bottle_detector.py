"""Main file of the project."""

from time import sleep, time

import cv2 as cv

from common import processing

REC_ON = True

def mainloop():
    """ Grap frames from both cameras and process them."""

    left_cam = cv.VideoCapture(2)
    if REC_ON:
        out_vid = cv.VideoWriter('output.mp4', 0x7634706d, 20.0, (640, 480))
    #left_cam.set(cv.CAP_PROP_FPS, 30)
    sleep(2)

    if not left_cam.isOpened():
        print("Can't open left camera!")
        return

    frames_cnt = 0
    timer_left = time()

    while True:
        ret, frame = left_cam.read()
        if not ret:
            continue

        frames_cnt += 1

        cv.imshow('video-org', frame)
        processing.process_frame(frame)
        cv.imshow('video-proc', frame)
        
        #sleep(0.5)
        if frames_cnt >= 30:
            print("FPS: %f" % round(frames_cnt / (time() - timer_left), 2))
            frames_cnt = 0
            timer_left = time()

        # save demo
        if REC_ON:
            out_vid.write(frame)

        if cv.waitKey(1) == 27:
            break

    left_cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    mainloop()
