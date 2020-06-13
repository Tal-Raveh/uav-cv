# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:30:47 2019

@author: Tal
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./queries/view.jpg',0)          # queryImage
img2 = cv2.imread('./trains/map.jpg',0)            # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks = 50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#plt.imshow(img3,),plt.show()
#plt.xticks([]), plt.yticks([])    # to hide tick values on X and Y axis
cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
cv2.imshow('matching',img3)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('./results/res5.png',img3)

cv2.destroyAllWindows()