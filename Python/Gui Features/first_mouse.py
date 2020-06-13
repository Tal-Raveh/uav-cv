# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:02:31 2018

@author: Tal
"""

import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),20,(0,255,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Click me!')
cv2.setMouseCallback('Click me!',draw_circle)

while(1):
    cv2.imshow('Click me!',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()