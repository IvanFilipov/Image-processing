from collections import Counter

import numpy             as np
import cv2               as cv
import matplotlib.pyplot as plt

CUBE_IMG = "../res/cube.jpg"

def solve_task1():
    m = np.random.randint(256, size=100)
    hist = Counter(m)

    plt.bar(hist.keys(), hist.values())
    plt.show()

def solve_task2():
    cdf = np.random.randint(256, size=100)
    hist = Counter(cdf)
    cdf[0] = hist[0]
    
    for i in range(1, cdf.size):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = np.min(cdf[cdf > 0])
    m = cdf.size - 1
    l = 256
    eq_hist = np.zeros(l, dtype=int)
    eq_hist = np.round((cdf - cdf_min) / m * (l -1)).astype(np.int)

    print (eq_hist)
    print (cdf[eq_hist])
    plt.figure(2)
    plt.plot(eq_hist)
    plt.show()

def solve_task5():
    img = cv.imread(CUBE_IMG, cv.IMREAD_GRAYSCALE)
    cv.imshow("image", img)

def solve_task6():
    cube_mat_clr  = cv.imread(CUBE_IMG, cv.IMREAD_COLOR)
    cube_mat_gray  = cv.cvtColor(cube_mat_clr, cv.COLOR_BGR2GRAY)
    cube_mat_hsv   = cv.cvtColor(cube_mat_clr, cv.COLOR_BGR2HSV)
    cv.imshow("cube_Gray", cube_mat_gray)
    cv.imshow("cube_HSV",  cube_mat_hsv)

if __name__ == '__main__':
    solve_task6()
    cv.waitKey(0)
    cv.destroyAllWindows()