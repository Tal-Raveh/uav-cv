"""
_ ~ Camera locator ~ _
    Uses ORB to match features between the camera frame and a satellite map
@author: Tal Raveh
"""

import numpy as np
# import sympy as sp
import cv2
import rasterio
import gps_utils
import pickle
import time
import random

gps_map = gps_utils.GPS_utils()

# %%
#   Finds the camera location for a single view picture
def single_pic():
    map_img = cv2.imread('./map/oneshot.jpg')
    view_img = cv2.imread('./view/DJI_0579.JPG')

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
    map_name = './map/oneshot_calibrated_modified'
    map_img = cv2.imread(map_name + '.tif')
    view_map = cv2.imread(map_name + '.tif')
    capture = cv2.VideoCapture('./view/circulation.mp4')
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = frame_count/fps
    
    # Initiate ORB detector
    #                   (nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_HARRIS_SCORE, patchSize = 31, fastThreshold = 20)
    orb = cv2.ORB_create(nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 50, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_HARRIS_SCORE, patchSize = 100, fastThreshold = 21)
    # find the keypoints and descriptors with ORB
    kp_t,des_t = orb.detectAndCompute(map_img , None)
    
    n_match = 17
    frame_skip = (10+0.1)/fps # 1/fps doesn't skip any frame
    """
    div1 = 1    # smaller divider (v axis)
    div2 = 0    # larger divider (u axis)
    for i in range(round(n_match/2),1,-1):
        if (n_match % i == 0 and div1 > div2):
            div1 = i
            div2 = round(n_match/i)
    """
    rc_zero = np.array([[],[],[]])
    
    watch = input("What would you like to watch? Choose your decision:\n \
    \t[1] Clean frame of the Pilot view and the map in different window.\n \
    \t[2] Matching results include the Pilot view and the Satellite map.\n \
    \t[else] None.\n Your answer: ")
    
    print("Simulation has started, press 'p' any time to pause.\n")
    s_time = time.time()
    p_time = 0
    while(1): #capture.isOpened()
        ret, view_frame = capture.read()
        if ret:
            s_p_time = 0
            # Press Q on keyboard to  exit    
            if (cv2.waitKey(1) & 0xFF == ord('p')):
                s_p_time = time.time()
                print("Simulation paused, press any key to continue or 'q' to stop and quit.\n")
                k = cv2.waitKey(0) & 0xFF
                if (k == ord('q')):
                    break
            if s_p_time:
                p_time += time.time()-s_p_time
            c_time = time.time()-s_time-p_time
            c_runtime = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            
            
            if (c_runtime%frame_skip <= 1/fps):
                print("-> Running time: %.2f/%.2f    [%.2f%%]\t(Real time elapsed: %.4f)" %(c_runtime , duration , (c_runtime*100)/duration , c_time))
                list_q , list_t , img_out = matcher(view_frame , map_img , n_match , orb , kp_t,des_t)
                rc_0 , rc_pix = locator(list_q , list_t , map_name)
                rc_zero = np.hstack((rc_zero , rc_0))
            else:
                print("Running time: %.2f/%.2f    [%.2f%%]\t(Real time elapsed: %.4f)" %(c_runtime , duration , (c_runtime*100)/duration , c_time))
            
            if (watch == '1'):
                cv2.namedWindow('Pilot View', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('Pilot View', view_frame.shape[1], view_frame.shape[0])
                cv2.imshow('Pilot View',view_frame)
                cv2.namedWindow('Map View', cv2.WINDOW_NORMAL)
                cv2.circle(view_map,(rc_pix[0],rc_pix[1]),25,(255,100,0),-1)
                cv2.circle(view_map,(rc_pix[0],rc_pix[1]),20,(100,0,0),-1)
                cv2.circle(view_map,(rc_pix[0],rc_pix[1]),5,(255,255,0),-1)
                cv2.imshow('Map View',view_map)
            elif (watch == '2'):
                cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),30,(255,100,0),-1)
                cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),25,(100,0,0),-1)
                cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),5,(255,255,0),-1)
                cv2.namedWindow('Pilot View | Satellite Map', cv2.WINDOW_NORMAL)
                cv2.imshow('Pilot View | Satellite Map', img_out)
        else: 
            break       # the video got to EOF
    
    print("Video frame per second (fps): %.4f [fps]" %(fps))
    print("Running fps: %.4f [fps]" %(capture.get(cv2.CAP_PROP_POS_FRAMES)/c_time))
    print("Average matching time: %.4f [sec]\t(1/fps=%.4f [sec])" %(c_time/(c_runtime/frame_skip) , 1/fps))
    capture.release()
    
    cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),30,(255,100,0),-1)
    cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),25,(100,0,0),-1)
    cv2.circle(img_out,(view_frame.shape[1]+rc_pix[0],rc_pix[1]),5,(255,255,0),-1)
    # watching and saving option
    cv2.namedWindow('matching', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('matching', img_out.shape[1], int(img_out.shape[0]/2))
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
def matcher(query , train , n_match , orb , kp_t,des_t , div1=1,div2=1):
    """
    # Initiate ORB detector
    orb = cv2.ORB_create()#nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 50, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_HARRIS_SCORE, patchSize = 100, fastThreshold = 15)
    # find the keypoints and descriptors with ORB
    kp_t,des_t = orb.detectAndCompute(train , None)
    """
    """
    # create BFMatcher object
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
    
    banana = []
    list_t = []
    list_q = []
    img_out = 0
    
    q_size = query.shape
    for i in range(div1):
        for j in range(div2):
            mask = np.zeros((q_size[0],q_size[1]),dtype = np.uint8)
            mask[round((q_size[0]/div1)*i):round((q_size[0]/div1)*(i+1)) , round((q_size[1]/div2)*j):round((q_size[1]/div2)*(j+1))] = 255
            
            kp_q,des_q = orb.detectAndCompute(query , mask)
            # Match descriptors
            matches = bfm.match(des_q , des_t)
            # Sort them in the order of their distance
            matches_sort = sorted(matches , key = lambda x:x.distance)
            
            banana.append(matches_sort[0])
            
            list_q.append(kp_q[matches_sort[0].queryIdx].pt)
            list_t.append(kp_t[matches_sort[0].trainIdx].pt)
            
    """
    
    kp_q,des_q = orb.detectAndCompute(query , None)
    # create BFMatcher object
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
    # Match descriptors
    matches = bfm.match(des_q , des_t)
    # Sort them in the order of their distance
    matches_sort = sorted(matches , key = lambda x:x.distance)
    """
    # Draw first n_match matches
    img_out = cv2.drawMatches(query , kp_q , train , kp_t ,
                              matches_sort[:n_match] , None , [255,0,0] , flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)    
    """       
    banana = matches_sort[:n_match]
    """
    flag = 1
    pois = 0
    i = 0
    banana = []
    while (pois < n_match):
        mat_i = matches_sort[i]
        current_pt = kp_q[mat_i.queryIdx].pt
        for mat_j in banana:
            prev_pt = kp_q[mat_j.queryIdx].pt
            dist = np.sqrt((prev_pt[0]-current_pt[0])**2 + (prev_pt[1]-current_pt[1])**2)
            if (dist < query.shape[0]/6):
                flag = 0
                break
            else:
                flag = 1
        if flag:
            list_q.append(kp_q[mat_i.queryIdx].pt)
            list_t.append(kp_t[mat_i.trainIdx].pt)
            banana.append(mat_i)
            pois += 1
        i += 1
    """
    
    src_pts = np.float32([ kp_q[m.queryIdx].pt for m in matches_sort[:n_match]]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_t[m.trainIdx].pt for m in matches_sort[:n_match]]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
    matchesMask = mask.ravel().tolist()
    h,w = query.shape[:2]
    pts = np.float32([ [1,1],[1,h-2],[w-2,h-2],[w-2,1] ]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts,M)
    dst += (w, 0)  # adding offset
    
    draw_params = dict(matchColor = (0,0,255), # draw matches in red color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    img_out = cv2.drawMatches(query,kp_q,train,kp_t,banana, None,**draw_params)
    
    # Draw bounding box in green
    img_out = cv2.polylines(img_out, [np.int32(dst)], True, (0,255,0),10, cv2.LINE_AA)
    
    """
    # get the first n_match matches coordinates
    list_q = []
    list_t = []
    
    for mat in matches_sort[:n_match]:
        list_q.append(kp_q[mat.queryIdx].pt)
        list_t.append(kp_t[mat.trainIdx].pt)
    """
    
    # Choose your own points
    # Random
    pts_q = np.float32([ [random.randint(w/6,5*w/6),random.randint(h/4,3*h/4)] for i in range(12) ]).reshape(-1,1,2)
    """
    # 12:
    pts_q = np.float32([ [w/4,h/3]      ,[5*w/12,h/3-20]      ,[2*w/3,h/3-5]       ,[3*w/4,h/3-10]     ,
                         [w/4-15,5*h/9] ,[5*w/12+7.5,5*h/9+30],[2*w/3+15,5*h/9-30] ,[3*w/4+7.5,5*h/9]  ,
                         [w/4-7.5,2*h/3],[5*w/12+15,2*h/3+5]  ,[2*w/3+7.5,2*h/3+20],[3*w/4+15,2*h/3+10] ]).reshape(-1,1,2)
    
    # 6:
    pts_q = np.float32([ [2*w/9,h/3]     , [4*w/9,h/2-20]  , [7*w/9,h/3+20]      ,
                         [2*w/9+20,2*h/3], [5*w/9.5,h/2+20], [7*w/9-20,2*h/3-20] ]).reshape(-1,1,2)
    """
    pts_t = cv2.perspectiveTransform(pts_q,M)
    
    list_q = pts_q.ravel().tolist()
    list_q = list(zip(list_q[::2] , list_q[1::2]))
    
    list_t = pts_t.ravel().tolist()
    list_t = list(zip(list_t[::2] , list_t[1::2]))
    """
    cv2.namedWindow('Pilot View | Satellite Map', cv2.WINDOW_NORMAL)
    cv2.imshow('Pilot View | Satellite Map', img_out)
    k = cv2.waitKey(0) & 0xFF
    #cv2.destroyAllWindows()
    """
    return list_q , list_t , img_out

# %% 
#   Finds the camera location vector using the matched points
#   list_q = list of pixeled vectors from the image represents the view frame
#   list_t = list of pixeled vectors from the image represents the satellite map
def locator(list_q , list_t , map_name):
    # map vector - 3D ~ from satellite map
    rp_map = get_map_cords(list_t,map_name)
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
    
    """
    # Lens matrix - K
    K = sp.Matrix(np.reshape(sp.symbols('k1:4(1:5)'),(3,4)))
    K[:,-1] = np.zeros((3,1))
    K[1:3,0] = np.zeros((2,1))
    K[-1,1] = 0.0
    K[-1,2] = 1.0
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
    # use DLT to find the general camera matrix - P
    P = dlt(mrp_map , mrp_tag)
    
    H_inf = np.matrix(P[:,0:3], dtype=np.float_)
    h_hat = np.array(P[:,3] , ndmin=2 , dtype=np.float_).T
    
    # The camera position
    rc_enu = np.array(-H_inf.I*h_hat)
    """
    # Define camera matrix K
    K = np.array([[1206.8854, 0., 960.0],
                  [0., 1206.8854, 540.0],
                  [0., 0., 1.]])
    
    K = np.array([[4479.94, 0., 2727.48],
              [0., 4374.51, 1811.92],
              [0., 0., 1.]])
    
    d = np.array([-0.0196948, 0.00652119, 0.000237354, 0.00108494, 0.0909295])
    
    #h,w = (1080,1920)
    #newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), alpha=0.2)
    
    #_, rvec, tvec = cv2.solvePnP(rp_map.reshape(-1,1,3) , rp_tag.reshape(-1,1,2) , K , d , flags=cv2.SOLVEPNP_AP3P)
    
    _, rvec, tvec, _ = cv2.solvePnPRansac(rp_map.reshape(-1,1,3) , rp_tag.reshape(-1,1,2) , K , d , iterationsCount=100 , reprojectionError=8.0 , flags=cv2.SOLVEPNP_EPNP)
    
    R,_ = cv2.Rodrigues(rvec)
    
    rc_enu = np.dot(-R.T,tvec)
    
    _______________
    
    # Find R and K matrices
    q,r = np.linalg.qr(H_inf.I)
    R = np.matrix(q).I
    K = np.matrix(r).I
    K = (1/K[2,2])*K
    A = np.vstack((np.hstack((R,-rc_enu)),np.array([0,0,0,1])))
    F = np.matrix([[0.010,0,0,0],[0,0.010,0,0],[0,0,1,0]])
    C = K*F*A
    cameraMatrix,rotMatrix,transVect,_,_,_,_ = cv2.decomposeProjectionMatrix(P)
    """
    rc_geo = gps_map.enu2geo(rc_enu[0],rc_enu[1],rc_enu[2])
    return np.array([rc_geo[1],rc_geo[0],rc_geo[2]],ndmin=2) , get_pixel_cords(rc_geo,map_name)

# %%
#   Finds the points location on the global map and returns their 3D vectors
#   list_t = list of pixeled vectors from the image represents the satellite map
def get_map_cords(list_t , map_name):
    rp_map = np.array([[],[],[]],dtype=np.float_)
    with open(map_name + '.tfw' , "r") as map_file:
        with rasterio.open('ASTGTM2_N31E034_dem.tif') as src:
            map_data = [float(line) for line in map_file]
            gps_map.setENUorigin(map_data[5] , map_data[4] , 0.0)
            # map_data[0] = A: pixel size in the x-direction in map units/pixel
            # map_data[1] = D: rotation about y-axis
            # map_data[2] = B: rotation about x-axis
            # map_data[3] = E: pixel size in the y-direction in map units, almost always negative
            # map_data[4] = C: x-coordinate of the center of the upper left pixel
            # map_data[5] = F: y-coordinate of the center of the upper left pixel
            for rp in list_t:
                long = np.float(map_data[0]*rp[0]) + np.float(map_data[2]*rp[1]) + map_data[4]     # A*px+B*py+C
                lat = np.float(map_data[1]*rp[0]) + np.float(map_data[3]*rp[1]) + map_data[5]      # D*px+E*py+F
                try:
                    alt = [float(val[0]) for val in src.sample([(long,lat)])]
                except:
                    print("An exception occurred - point not in map: (lo %.6f,la %.6f,al ???)" %(long,lat))
                rp_map = np.hstack((rp_map , np.array(gps_map.geo2enu(lat , long , alt[0]) , ndmin=2)))
    
    return rp_map

# %%
#   Finds a point location on the global map and returns its pixel vector
#   rp_geo = point of 3D vector located in the image represents the satellite map
def get_pixel_cords(rp_geo , map_name):
    rp_pix = np.array([[],[],[]],dtype=np.float_)
    with open(map_name + '.tfw' , "r") as map_file:
        map_data = [float(line) for line in map_file]
        # map_data[0] = A: pixel size in the x-direction in map units/pixel
        # map_data[1] = D: rotation about y-axis
        # map_data[2] = B: rotation about x-axis
        # map_data[3] = E: pixel size in the y-direction in map units, almost always negative
        # map_data[4] = C: x-coordinate of the center of the upper left pixel
        # map_data[5] = F: y-coordinate of the center of the upper left pixel
        a1 = np.array([[map_data[0]],[map_data[1]]])
        a2 = np.array([[map_data[2]],[map_data[3]]])
        b = np.array([[rp_geo[1,0]-map_data[4]],[rp_geo[0,0]-map_data[5]]])
        px = (b[0]*a2[1]-b[1]*a2[0])/(a1[0]*a2[1]-a1[1]*a2[0])
        py = (a1[0]*b[1]-a1[1]*b[0])/(a1[0]*a2[1]-a1[1]*a2[0])
        rp_pix = np.array([px,py],ndmin=2)
    
    return rp_pix

# %%
#   A = DLT(x,y) solves for the projective transformation matrix A with respect to
#   the linear system Ax ~ y where ~ denotes equality up to a scale, using the
#   Direct Linear Transformation technique. A is a m-by-n matrix, x is a n-by-kx
#   matrix that contains kx source points in column vector form and y is a m-by-ky
#   matrix containning ky target points in column vector form. The solution is
#   normalised as any multiple of A also satisfies the equation.
def dlt(x , y):
    n,kx = np.shape(x)
    m,ky = np.shape(y)
    # Dimensions check:
    if (kx != ky):
        print("Bad matrices input: the dimensions of x and y matrices doesn't fit!")
        return
    # Create the flat matrix from x and y
    M =  np.zeros((2*kx,n*m))
    for i in range(kx):
        M[2*i,:] = np.array(np.hstack((-x[:,i],np.zeros(n),x[:,i]*y[0,i])),ndmin=2)
        M[2*i+1,:] = np.array(np.hstack((np.zeros(n),-x[:,i],x[:,i]*y[1,i])),ndmin=2)
    _,_,vh = np.linalg.svd(M)
    # The solution minimising |M*C| is the right singular vector
    # corresponding to the smallest singular value
    #print("||M*C|| = %.6g" %np.linalg.norm(np.dot(M,vh[-1,:])))
    A = np.reshape(vh[-1,:] , (m,n))
    return A
    """
    # Make y inhomogeneous:
    r = np.setdiff1d(range(m),m-1)
    y = y[r]/y[m-1]
    # Build the homogeneous linear system:
    A = np.zeros((m , n , kx , m-1))
    for i in r:
        g = -x*y[i]
        A[i,:,:,i] = np.reshape(x , (1,n,kx) , order='F')
        A[m-1,:,:,i] = np.reshape(g , (1,n,kx) , order='F')
    # Convert to a big flat matrix
    A = A.reshape(m*n , -1 , order='F')
    # Solve the homogeneous linear system using SVD
    _,_,vh = np.linalg.svd(np.dot(A , A.T))
    # The solution minimising |A.T*A| is the right singular vector
    # Corresponding to the smallest singular value
    print("||M*C|| = " + str(np.linalg.norm(np.dot(np.dot(A , A.T),vh[-1,:]))))
    A = np.reshape(vh.T[:,-1] , (m,n) , order='F')
    # Some normalisation (optional):
    A = A / np.linalg.norm(A) * np.sign(A[0,0])
    return A
    """

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
