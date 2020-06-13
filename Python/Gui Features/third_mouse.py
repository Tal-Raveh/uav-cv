# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:53:25 2018

@author: Tal
"""

import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('color')

# create trackbars for color change
cv2.createTrackbar('R','color',0,255,nothing)
cv2.createTrackbar('G','color',0,255,nothing)
cv2.createTrackbar('B','color',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF\n1 : ON'
cv2.createTrackbar(switch,'color',0,1,nothing)

while(1):
    cv2.imshow('color',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','color')
    g = cv2.getTrackbarPos('G','color')
    b = cv2.getTrackbarPos('B','color')
    s = cv2.getTrackbarPos(switch,'color')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()