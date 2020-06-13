import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define camera matrix K
K = np.array([[4479.94, 0., 2727.48],
              [0., 4374.51, 1811.92],
              [0., 0., 1.]])

# Define distortion coefficients d
d = np.array([-0.0196948, 0.00652119, 0.000237354, 0.00108494, 0.0909295])

# Read an example image and acquire its size
img = cv2.imread("C:\Tal's DATA\Testim\BGU Campus\BackYard\DJI_0008.JPG")
h, w = img.shape[:2]

# Generate new camera matrix from parameters
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)

# Generate look-up tables for remapping the camera image
mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

# Remap the original image to a new image
newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Display old and new image
fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(img)
oldimg_ax.set_title('Original image')
newimg_ax.imshow(newimg)
newimg_ax.set_title('Unwarped image')
plt.show()