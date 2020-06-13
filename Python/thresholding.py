# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:18 2019

@author: Tal
"""

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,555,2)

titles = ['Original Image', 'Global Thresholding',
          'Adaptive Gaussian Thresholding']
images = [img, th1, th2]

for i in range(3):
    plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()