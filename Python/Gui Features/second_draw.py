# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:31:52 2018

@author: Tal
"""

import cv2
import numpy as np

drawing = False # true if left mouse is pressed
erasing = False # true if right mouse is pressed

def nothing(x):
    pass

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing,erasing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img,(x,y),s,(b,g,r),-1)
        elif erasing:
            cv2.circle(img,(x,y),s,(0,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False

# Create a black image, a window
img = np.zeros((400,512,3), np.uint8)
cv2.namedWindow('paint')
cv2.setMouseCallback('paint',draw_circle)

# create trackbars for color change
cv2.createTrackbar('R','paint',0,255,nothing)
cv2.createTrackbar('G','paint',0,255,nothing)
cv2.createTrackbar('B','paint',0,255,nothing)
cv2.createTrackbar('Size','paint',1,100,nothing)

while(1):
    cv2.imshow('paint',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','paint')
    g = cv2.getTrackbarPos('G','paint')
    b = cv2.getTrackbarPos('B','paint')
    s = cv2.getTrackbarPos('Size','paint')

cv2.destroyAllWindows()