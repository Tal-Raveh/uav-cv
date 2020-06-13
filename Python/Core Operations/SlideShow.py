# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:37:07 2018

@author: Tal
"""

import cv2
import numpy as np

img0 = cv2.imread('CaptureBGU.png')
img1 = cv2.imread('CaptureMENG.png')
img2 = cv2.imread('Batman.png')
img3 = cv2.imread('OpenCV_Logo.png')
img4 = cv2.imread('wolverinSA.jpg')

for x in range(4):
    one = eval('img{}'.format(x))
    two = eval('img{}'.format(x+1))
    cv2.imshow('kaki',one)
    cv2.waitKey(0)
    if one.size == two.size:
        dst = cv2.addWeighted(one,0.5,two,0.5,0)
    cv2.imshow('kaki',dst)
    cv2.waitKey(0)
    cv2.imshow('kaki',two)
    cv2.waitKey(0)

cv2.destroyAllWindows()