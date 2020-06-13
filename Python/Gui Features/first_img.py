# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image
img = cv2.imread('DSC_5792.jpg',1)

img2 = img[:,:,::-1]
plt.subplot(121);plt.imshow(img);plt.title('BGR image')  # expects distorted color
plt.xticks([]), plt.yticks([])    # to hide tick values on X and Y axis
plt.subplot(122);plt.imshow(img2);plt.title('RGB image') # expect true color
plt.xticks([]), plt.yticks([])    # to hide tick values on X and Y axis
plt.show()

cv2.namedWindow('BGR image', cv2.WINDOW_NORMAL)
cv2.imshow('BGR image',img)  # expects true color
cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
cv2.imshow('RGB image',img2) # expects distorted color
"""
plt.imshow(img, cmap = 'cool', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""
"""
cv2.namedWindow('Family portrait', cv2.WINDOW_NORMAL)
cv2.imshow('Family portrait',img)
"""
k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('familygray.png',img)
    cv2.destroyAllWindows()