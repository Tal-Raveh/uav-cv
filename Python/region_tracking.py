# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 22:36:00 2019

@author: Tal
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# define range of green color in HSV
lower_green = np.array([50,50,50])
upper_green = np.array([70,255,255])

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('frame3.jpg',frame)
        cv2.imwrite('color_f3.jpg',res)
        break

cap.release()
cv2.destroyAllWindows()