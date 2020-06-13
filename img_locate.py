# -*- coding: utf-8 -*-
"""
_ ~ Camera locator ~ _
    Uses ORB to match features between the camera frame and a satellite map
@author: Tal Raveh
"""

import numpy as np
#import sympy as sp
import cv2
import rasterio
import pickle
import time

# %%
#   Finds the camera location for a single view picture
def single_pic():
    map_img = cv2.imread('./map/map_1.jpg')
    view_img = cv2.imread('./view/single_view.jpg')

    s_time = time.time()
    
    list_q , list_t , img_out = matcher(view_img , map_img , 6)
    
    rc_zero = locator(list_q , list_t)
    
    c_time = time.time()-s_time
    print("Running time for a single picture: %.4f [sec]" %(c_time))
    
    # watching and saving option
    cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('matching', img_out.shape[1], int(img_out.shape[0]/2))
    cv2.imshow('matching',img_out)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if k == ord('s'):
        choice = input("You chose to save the results. Choose your decision:\n \
        \t[1] Save both result image and location data.\n \
        \t[2] Save result image only.\n \
        \t[3] Save location data only.\n \
        \t[else] Save none.\n Your answer: ")
        if (choice == '1') or (choice == '2'):
            img_name = input("Please insert destination file name (result image): ")
            cv2.imwrite("./result/" + img_name + ".jpg",img_out)
        if (choice == '1') or (choice == '3'):
            res_name = input("Please insert destination file name (location data): ")
            with open("./result/" + res_name + '.loc' , "wb") as res_file:
                pickle.dump(rc_zero , res_file)
            # with open('result.loc' , "rb") as res_file:
            #     location = pickle.load(res_file)
    
    return rc_zero

# %%
# Finds the camera location for a multi view video
def multi_vid():
    map_img = cv2.imread('./map/map_1.jpg')
    capture = cv2.VideoCapture('./view/lookatme.mp4')
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = frame_count/fps
    
    rc_zero = np.array([[],[],[]])
    
    watch = input("What would you like to watch? Choose your decision:\n \
    \t[1] Clean frame of the Pilot view.\n \
    \t[2] Matching results include the Pilot view and the Satellite map.\n \
    \t[else] None.\n Your answer: ")
    
    print("Simulation has started, press 'p' any time to pause.\n")
    s_time = time.time()
    while(capture.isOpened()):
        ret, view_frame = capture.read()
        if ret:
            c_runtime = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            c_time = time.time()-s_time
            print("Running time: %.2f/%.2f    [%.2f%%]\t(Real time elapsed: %.4f)" %(c_runtime , duration , (c_runtime*100)/duration , c_time))
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('p'):
                print("Simulation paused, press any key to continue or 'q' to stop and quit.\n")
                k = cv2.waitKey(0) & 0xFF
                if (k == ord('q')):
                    break

            list_q , list_t , img_out = matcher(view_frame , map_img , 6)
            
            if (watch == '1'):
                cv2.namedWindow('Pilot View', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Pilot View', view_frame.shape[1], view_frame.shape[0])
                cv2.imshow('Pilot View',view_frame)
            elif (watch == '2'):
                cv2.namedWindow('Pilot View | Satellite Map', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Pilot View | Satellite Map', img_out.shape[1], int(img_out.shape[0]/2))
                cv2.imshow('Pilot View | Satellite Map', img_out)
            
            rc_zero = np.hstack((rc_zero , locator(list_q , list_t)))
        else: 
            break       # the video got to EOF
    
    capture.release()
    print("Video seconds per frame (1/fps): %.4f [sec]" %(1/fps))
    print("Running time for a single frame: %.4f [sec]" %(c_time/rc_zero.shape[1]))
    
    # watching and saving option
    cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('matching', img_out.shape[1], int(img_out.shape[0]/2))
    cv2.imshow('matching',img_out)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if k == ord('s'):
        choice = input("You chose to save the results. Choose your decision:\n \
        \t[1] Save both result image and location data.\n \
        \t[2] Save result image only.\n \
        \t[3] Save location data only.\n \
        \t[else] Save none.\n Your answer: ")
        if (choice == '1') or (choice == '2'):
            img_name = input("Please insert destination file name (result image): ")
            cv2.imwrite("./result/" + img_name + ".jpg",img_out)
        if (choice == '1') or (choice == '3'):
            res_name = input("Please insert destination file name (location data): ")
            with open("./result/" + res_name + '.loc' , "wb") as res_file:
                pickle.dump(rc_zero , res_file)
            # with open('result.loc' , "rb") as res_file:
            #     location = pickle.load(res_file)
    
    return rc_zero

# %%
#   Feature matching using ORB – Oriented FAST and Rotated BRIEF
#   query = an image represents the view frame
#   train = an image represents the satellite map
#   n_match = number of matches to locate (POI - Points of Interest)
def matcher(query , train , n_match):
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

# %% 
#   Finds the camera location vector using the matched points
#   list_q = list of pixeled vectors from the image represents the view frame
#   list_t = list of pixeled vectors from the image represents the satellite map
def locator(list_q , list_t):
    # map vector - 3D ~ from satellite map
    rp_map = get_map_cords(list_t)
    # pixel vector - 2D ~ from camera
    rp_tag = np.array(list_q,dtype=np.float_).T
    """
    obj = np.ascontiguousarray(rp_map.T[:,:3]).reshape((11,1,3))
    img = np.ascontiguousarray(rp_tag.T[:,:2]).reshape((11,1,2))
    ret, rvecs, tvecs = cv2.solvePnP(obj, img, np.array([[1,0,1376/2],[0,1,776/2],[0,0,1]]),None)
    R_ans = np.matrix(cv2.Rodrigues(rvecs)[0])
    T_ans = tvecs
    return np.array(np.dot(-R_ans.I , T_ans))
    """
    # add ones at the bottom of each vector
    mrp_map = np.vstack([rp_map,np.ones(np.shape(rp_map)[1])])
    mrp_tag = np.vstack([rp_tag,np.ones(np.shape(rp_tag)[1])])
    
    # use DLT to find the camera matrix - C
    C = dlt(mrp_map , mrp_tag)
    """
    # Lens matrix - K
    K = np.array([[1    ,0  ,1376/2 ,0],
                  [0    ,1  ,776/2  ,0],
                  [0    ,0  ,1      ,0]])
    # Kinematic transformation matrix - A
    R = sp.Matrix(np.reshape(sp.symbols('r1:4(1:4)'),(3,3)))
    T = sp.Matrix(np.reshape(sp.symbols('t1:4'),(3,1)))
    A = sp.Matrix.vstack(sp.Matrix.hstack(R,T),sp.Matrix([0,0,0,1]).T)
    ~~~
    K * A =
    [[k11*r11 + k13*r31, k11*r12 + k13*r32, k11*r13 + k13*r33, k11*t1 + k13*t3],
     [k22*r21 + k23*r31, k22*r22 + k23*r32, k22*r23 + k23*r33, k22*t2 + k23*t3],
     [              r31,               r32,               r33,              t3]]
    ~~~
    # Solve the equations
    sol = sp.solve(K*A-C,A)
    # The results
    R_ans = np.matrix(np.zeros((3,3)) , dtype=np.float_)
    T_ans = np.zeros((3,1) , dtype=np.float_)
    for i in range(3):
        T_ans[i] = sol[T[i]]
        for j in range(3):
            R_ans[i,j] = sol[R[i,j]]
    
    R_ans = np.matrix(np.zeros((3,3)) , dtype=np.float_)
    T_ans = np.zeros((3,1) , dtype=np.float_)
    # Solve the equations
    for i in range(3):
        R_ans[2,i] = C[2,i]             # r[3,1:3] (last row)
        for j in range(1,-1,-1):        # r[2:1,1:3] (2nd & 1st rows)
            R_ans[j,i] = (C[j,i] - K[j,2]*R_ans[2,i])/K[j,j]
    
    T_ans[2] = C[2,3]                   # t[3] (last value)
    for j in range(1,-1,-1):            # t[2:1] (2nd & 1st values)
        T_ans[j] = (C[j,3] - K[j,2]*T_ans[2])/K[j,j]
    """
    R_ans = np.matrix(C[:,0:3], dtype=np.float_)
    T_ans = np.array(C[:,3] , ndmin=2 , dtype=np.float_).T
    # The camera position
    return np.array(-R_ans.I*T_ans)

# %%
#   Finds the points location on the global map and returns their 3D vectors
#   list_t = list of pixeled vectors from the image represents the satellite map
def get_map_cords(list_t):
    rp_map = np.array([[],[],[]])
    with open('./map/map_1.jgw' , "r") as map_file:
        with rasterio.open('ASTGTM2_N31E034_dem.tif') as src:
            map_data = [float(line) for line in map_file]
            # map_data[0] = A: pixel size in the x-direction in map units/pixel
            # map_data[1] = D: rotation about y-axis
            # map_data[2] = B: rotation about x-axis
            # map_data[3] = E: pixel size in the y-direction in map units, almost always negative
            # map_data[4] = C: x-coordinate of the center of the upper left pixel
            # map_data[5] = F: y-coordinate of the center of the upper left pixel
            for rp in list_t:
                lat = map_data[4]+rp[0]*map_data[0]     # C+A*px
                long = map_data[5]+rp[1]*map_data[3]    # F+E*py
                alt = [float(val[0]) for val in src.sample([(lat,long)])]
                rp_map = np.hstack((rp_map , np.array([lat , long , alt[0]] , ndmin=2).T))
    
    return rp_map

# %%
#   A = DLT(x,b) solves for the projective transformation matrix A with respect to
#   the linear system Ax ~ b where ~ denotes equality up to a scale, using the
#   Direct Linear Transformation technique. A is a m-by-n matrix, x is a n-by-k
#   matrix that contains k source points in column vector form and b is a m-by-kb
#   matrix containning kb target points in column vector form. The solution is
#   normalised as any multiple of A also satisfies the equation.
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
    # A = A / np.linalg.norm(A) * np.sign(A[0,0])
    
    return A

# %%
#   run this function if it’s the main program
if (__name__ == "__main__"):
    selc = input("Select data input:\n \
    \t[1] single picture.\n \
    \t[2] Mullti frame video.\n \
    \t[else] Stop and leave me alone.\n Your answer: ")
    if (selc == '1'):
        location = single_pic()
        print("rc0 = {" + '\n'.join([''.join(['\t{}'.format(item) for item in row]) for row in location]) + " }")
    elif (selc == '2'):
        location = multi_vid()
