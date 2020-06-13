# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:30:47 2019

@author: Tal Raveh
"""

import cv2

img1 = cv2.imread('./queries/vieww.jpg',1)          # queryImage
img2 = cv2.imread('./trains/mapp.jpg',1)            # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors
matches = bf.match(des1,des2)
# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)
n_match = 3                                        # number of matches to show
# Draw first n_match matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,
                       matches[:(n_match)],None,(255,0,255),flags=2)
# get the first n_match matches coordinates
list_kp1 = []
list_kp2 = []
for mat in matches[n_match:(n_match+1)]:
    list_kp1.append(kp1[mat.queryIdx].pt)
    list_kp2.append(kp2[mat.queryIdx].pt)
# watching and saving option
cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
cv2.imshow('matching',img3)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('./results/resultt15.jpg',img3)

cv2.destroyAllWindows()