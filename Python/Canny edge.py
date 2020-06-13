# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:13:39 2019

@author: Tal
"""

import cv2

cap = cv2.VideoCapture(0)
m = 1;

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame1 = cv2.GaussianBlur(frame,(5,5),0)
    if ret == True:
        edges = cv2.Canny(frame1,55,155)

        # Display the resulting frame
        if m == 1:
            cv2.imshow('frame',edges)
        else:
            cv2.imshow('frame',frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('frame2.jpg',frame)
            cv2.imwrite('edge_f2.jpg',edges)
            break
        elif k == ord('m'):
            if m == 1:
                m = 0
            else:
                m = 1
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
