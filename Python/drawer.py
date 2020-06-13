# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2

# Create a black image
img = cv2.imread('./trains/mapp.jpg',1)

lineType = cv2.LINE_AA      # for nice curves
"""
# Draw a red circle inside the rectangle
img = cv2.circle(img,list_des2[0],4, (0,0,255),1,lineType)
"""
# Draw a blue half ellipse at the center
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1,lineType)

# Draw a yellow polygon with four vertices
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
"""
# Write 'OpenCV' at the bottom
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),3,lineType)
"""
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test',img)
"""
        # OpenCV Logo
# Create another black image
img2 = np.ones((512,512,3), np.uint8)
# Draw the red circle
img2 = cv2.circle(img2,(256,130),55,(0,0,255),42,lineType)
# Draw the green circle
img2 = cv2.circle(img2,(159,291),55,(0,255,0),42,lineType)
# Cut through the circles above
trit1 = np.array([[256,130],[159,291],[348,291]], np.int32)
trit1 = trit1.reshape((-1,1,2))
img2 = cv2.fillPoly(img2,[trit1],0,lineType)
# Draw the blue circle
img2 = cv2.circle(img2,(348,291),55,(255,0,0),42,lineType)
# Cut through the circle above
trit2 = np.array([[348,291],[297,211],[399,211]], np.int32)
trit2 = trit2.reshape((-1,1,2))
img2 = cv2.fillPoly(img2,[trit2],0,lineType)
# Write 'OpenCV' at the bottom
cv2.putText(img2,'OpenCV',(20,475), font, 4,(255,255,255),12,lineType)

cv2.imshow('OpenCV Logo',img2)
"""
k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('banana.jpg',img)

cv2.destroyAllWindows()