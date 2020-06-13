# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:30:47 2019

@author: Tal Raveh
"""

import numpy as np
import sympy as sp
import cv2


def main():
    view_path = './queries/view.jpg'
    map_path = './trains/map.jpg'
    list_q , list_t , img_out = matcher(view_path , map_path , 3)
    
    rc_zero = locator(list_q , list_t)
    
    print("rc0 = {" + '\n'.join([''.join(['\t{}'.format(item) for item in row]) for row in np.round(rc_zero,3)]) + " }")
    
    # the result should be close to - FROM GOOGLE EARTH
    rc_ggl = np.array([31.405000 , 34.421111  , 34       , 921       ]).T
    #                  ^Latitude   ^Longitude   ^Surface   ^Altitude
    
    # watching and saving option
    cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
    cv2.imshow('matching',img_out)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if k == ord('s'):
        print("You chose to save the result image. If you wish to cancel insert '0'")
        res_name = input("Please insert destination file name (0 to cancel): ")
        if (res_name != '0'):
            cv2.imwrite("./results/" + res_name + ".jpg",img_out)
    
    return rc_zero , rc_ggl


# queryImage
# trainImage
# number of matches to show
def matcher(view_path , map_path , n_match):
    query = cv2.imread(view_path , 1)
    train = cv2.imread(map_path , 1)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp_q,des_q = orb.detectAndCompute(query , None)
    kp_t,des_t = orb.detectAndCompute(train , None)
    # create BFMatcher object
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
    # Match descriptors
    matches = bfm.match(des_q , des_t)
    # Sort them in the order of their distance
    matches = sorted(matches , key = lambda x:x.distance)
    # Draw first n_match matches
    img_out = cv2.drawMatches(query , kp_q , train , kp_t ,
                              matches[:n_match] , None , [0,0,255] , flags=2)    
    # get the first n_match matches coordinates
    list_q = []
    list_t = []
    for mat in matches[:n_match]:
        list_q.append(kp_q[mat.queryIdx].pt)
        list_t.append(kp_t[mat.queryIdx].pt)
    
    return list_q , list_t , img_out


def locator(list_q , list_t):
    # map vector - 3D ~ from satellite map
    rp_map = np.array([[31.395701   ,31.405566  ,31.406478],
                       [34.414097   ,34.419729  ,34.411618],
                       [51          ,35         ,56]])
    # rp_map = np.array(list_q,dtype=np.float_).T
    # pixel vector - 2D ~ from camera
    rp_tag = np.array(list_t,dtype=np.float_).T
    # rp_tag = np.array([[271.226959228515625 ,830.103759765625   ,1042.6063232421875],\n[236.3904571533203125,647.95867919921875 ,306.063446044921875]])
    # add ones at the bottom of each vector
    mrp_map = np.vstack([rp_map,np.ones(np.shape(rp_map)[1])])
    mrp_tag = np.vstack([rp_tag,np.ones(np.shape(rp_tag)[1])])
    
    # use DLT to find the camera matrix - C
    C = dlt(mrp_map , mrp_tag)
    # Build matrix K
    K = np.array([[1    ,0  ,1366/2 ,0],
                  [0    ,1  ,767/2  ,0],
                  [0    ,0  ,1      ,0]])
    R = sp.Matrix(np.reshape(sp.symbols('r1:4(1:4)'),(3,3)))
    T = sp.Matrix(np.reshape(sp.symbols('t1:4'),(3,1)))
    A = sp.Matrix.vstack(sp.Matrix.hstack(R,T),sp.Matrix([0,0,0,1]).T)

    # Solve the equations
    sol = sp.solve(K*A-C,A)
    # The results
    R_ans = np.matrix(np.zeros((3,3)) , dtype=np.float_)
    T_ans = np.zeros((3,1) , dtype=np.float_)
    for i in range(3):
        T_ans[i] = sol[T[i]]
        for j in range(3):
            R_ans[i,j] = sol[R[i,j]]
    
    # The camera position
    rc_zero = np.array(np.dot(-R_ans.I , T_ans))
    
    return rc_zero


# A = DLT(x,b) solves for the projective transformation matrix A with respect to
# the linear system Ax ~ b where ~ denotes equality up to a scale, using the
# Direct Linear Transformation technique. A is a m-by-n matrix, x is a n-by-k
# matrix that contains k source points in column vector form and b is a m-by-kb
# matrix containning kb target points in column vector form. The solution is
# normalised as any multiple of A also satisfies the equation.
def dlt(x , b):
    n,k = np.shape(x)
    m,kb = np.shape(b)
    # Dimensions check:
    if (k != kb):
        print("Bad matrices input: the dimensions of x and b matrices doesn't fit!")
        return
    # Make b inhomogeneous:
    r = np.setdiff1d(range(m),m-1)
    b = b[r]/b[m-1]
    # Build the homogeneous linear system:
    A = np.zeros((m , n , k , m-1))
    for i in r:
        y = -x*b[i]
        A[i,:,:,i] = np.reshape(x , (1,n,k) , order='F')
        A[m-1,:,:,i] = np.reshape(y , (1,n,k) , order='F')
    # Convert to a big flat matrix
    A = A.reshape(m*n , -1 , order='F')
    # Solve the homogeneous linear system using SVD
    _,_,vh = np.linalg.svd(np.dot(A , A.T))
    # The solution minimising |A.T*A| is the right singular vector
    # Corresponding to the smallest singular value
    A = np.reshape(vh.T[:,-1] , (m,n) , order='F')
    # Some normalisation (optional):
    A = A / np.linalg.norm(A) * np.sign(A[0,0])
    
    return A


if (__name__ == "__main__"):
    location = main()
