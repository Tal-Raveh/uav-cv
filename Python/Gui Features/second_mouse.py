# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:16:40 2018

@author: Tal
"""

import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix+4,iy+4),(x-4,y-4),0,-1)
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),5)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix+4,iy+4),(x-4,y-4),(0,0,0),-1)
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),5)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
            
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('paint')
cv2.setMouseCallback('paint',draw_circle)

while(1):
    cv2.imshow('paint',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()