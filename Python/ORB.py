# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:09:23 2019

@author: Tal
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpg',0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)
plt.imshow(img2),plt.show()

cv2.namedWindow('new', cv2.WINDOW_NORMAL)
cv2.imshow('new',img2)  # expects true color

k = cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()