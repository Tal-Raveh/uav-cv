# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:55:54 2018

@author: Tal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('wolverinSA.jpg')

px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)

img[100,100] = [255,255,255]
print(img[100,100])
    
    #   BETTER
# accessing RED value
print(img.item(10,10,2))

# modifying RED value
img.itemset((10,10,2),100)
print(img.item(10,10,2))

# image properties (rows,columns,channels)
print(img.shape)    # for grayscale images, there will be no channels

# image total size (rows*columns*channels)
print(img.size)

# image data type @@@@@@@@@@@         IMPORTANT        @@@@@@@@@@@@@@
print(img.dtype)

face = img[652:725,337:404]
img[652:725,237:304] = face
img = cv2.rectangle(img,(337,652),(404,725),(0,0,255),1)
img = cv2.rectangle(img,(237,652),(304,725),(0,255,0),1)

cv2.namedWindow('memories',cv2.WINDOW_NORMAL)
cv2.imshow('memories',img)
    
# spliting and merging image channels (Heavy Runs)
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
# for one channel only:
b = img[:,:,0]
# black all the red!
img[:,:,2] = 0

cv2.imshow('memories',img)

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
    
BLUE = [255,0,0]

img1 = cv2.imread('OpenCV_Logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()