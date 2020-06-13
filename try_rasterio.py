# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:09:39 2019

@author: Tal
"""

import numpy as np
import rasterio

data = np.array([[],[],[]])
temp = np.array([])
with open("D:/Documents/אקדמיה/תואר שני - הנדסת מכונות/תיזה/workspace/result.txt" , "r") as file:
    for row in file:
        temp = [float(row[dig*3:(dig+1)*3-1]) for dig in range(14)]
        for item in row[1:]:
            temp = np.vstack((temp , float(item)))
        data = np.hstack((data , temp))
        

coords = ([(34.41607,31.40318)])
elevation = 'ASTGTM2_N31E034_dem.tif'

with rasterio.open(elevation) as src:
    [print(val[0]) for val in src.sample(coords)]
    #val is an array of values, 1 element 
                      #per band. src is a single band raster 
                      #so we only need val[0]