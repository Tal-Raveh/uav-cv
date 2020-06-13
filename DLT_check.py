import numpy as np
import rasterio
import gps_utils

gps_map = gps_utils.GPS_utils()

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
    print("||M*C|| = %.6g" %np.linalg.norm(np.dot(M,vh[-1,:])))
    A = np.reshape(vh[-1,:] , (m,n))
    y_new = np.dot(A,x)
    return A/np.average(y_new[-1])
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

if (__name__ == "__main__"):
    #C = np.array([[1,0,-400,40000],[0,-1,-300,30001],[0,0,-1,100]])
    rc_real = np.array([[34.83702163],[31.27669002],[430.]])
    
    gps_map.setENUorigin(31.27781694884 , 34.835577675976 , 0.0)
    """
    K = np.matrix([[-1.,0.,400.],
                   [0.,-1.,300.],
                   [0.,0.,1.]])
    """
    K = np.array([[4479.94, 0., 2727.48],
                  [0., 4374.51, 1811.92],
                  [0., 0., 1.]])
    F = np.matrix([[1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.]])
    R = np.matrix([[-1.,0.,0.],
                   [0.,1.,0.],
                   [0.,0.,-1.]])
    rc_c = np.array(-R.T*gps_map.geo2enu(rc_real[1] , rc_real[0] , rc_real[2]))
    
    A = np.vstack((np.hstack((R,rc_c)),np.array([0.,0.,0.,1.])))
    
    C = K*F*A
    
    """
    # Made up points (result ~ 1e-3 meters)
    x = np.array([[3.,1.,10.,2.,33.,0.],[2.,2.,0.,11.,12.,3.],[1.,3.,2.,2.5,0.,4.]])
    
    # GPS 5-10 points (result ~ 7.9e-6)
    x = np.array([[34.83617,34.83632,34.83652,34.83702,34.83712,34.83745],
                  [31.27666,31.27686,31.27679,31.27669,31.27657,31.27633],
                  np.zeros(6)])
    
    # MAP 5-10 points (result ~ 2.8e-5)
    x = np.array([[34.83617,34.83629,34.83651,34.83702,34.83712,34.83748],
                  [31.27668,31.27689,31.27681,31.27669,31.27658,31.27633],
                  np.zeros(6)])
    
    # GPS 2-7 points (result ~ 4.4e-5)
    x = np.array([[34.83590,34.83585,34.83597,34.83617,34.83632,34.83652],
                  [31.27613,31.27606,31.27685,31.27666,31.27686,31.27679],
                  np.zeros(6)])
    
    # MAP 2-7 points (result ~ 7.3e-4)
    x = np.array([[34.83590,34.83585,34.83597,34.83617,34.83629,34.83651],
                  [31.27613,31.27604,31.27685,31.27668,31.27689,31.27681],
                  np.zeros(6)])
    
    ## GOOD Comparison
    # GPS 4,8,11,15,16,19 points (result ~ 4.2e-6)
    x = np.array([[34.83597,34.83702,34.83797,34.83818,34.83714,34.83649],
                  [31.27685,31.27669,31.27596,31.27714,31.27755,31.27766],
                  np.zeros(6)])
    
    # MAP 4,8,11,15,16,19 points (result ~ 1.2e-5)
    x = np.array([[34.83597186,34.83702163,34.83796971,34.83817906,34.83714126,34.83648987],
                  [31.27684853,31.27669002,31.27596147,31.27713864,31.27755017,31.27765904],
                  np.zeros(6)])
    
    ## BAD Comparison
    # GPS 1,6,7,10,12,21 points (result ~ 7.3e-6)
    x = np.array([[34.83574,34.83632,34.83652,34.83745,34.83830,34.83648],
                  [31.27618,31.27686,31.27679,31.27633,31.27583,31.27710],
                  np.zeros(6)])
    
    # MAP 1,6,7,10,12,21 points (bad result ~ 5.7e-5)
    x = np.array([[34.83576849,34.83629427,34.83651499,34.83747503,34.83827955,34.83646833],
                  [31.27617381,31.27688681,31.27681264,31.27632574,31.27580774,31.27713744],
                  np.zeros(6)])
    
    ## Multple Comparison
    # GPS 1,6,7,10,12,21,4,8,11,15,16,19 points (good result ~ 2.3e-7)
    x = np.array([[34.83574,34.83632,34.83652,34.83745,34.83830,34.83648,34.83597,34.83702,34.83797,34.83818,34.83714,34.83649],
                  [31.27618,31.27686,31.27679,31.27633,31.27583,31.27710,31.27685,31.27669,31.27596,31.27714,31.27755,31.27766],
                  np.zeros(12)])
    """
    # MAP 1,6,7,10,12,21,4,8,11,15,16,19 points (good results ~ 3.4e-7)
    x = np.array([[34.83576849,34.83629427,34.83651499,34.83747503,34.83827955,34.83646833,34.83597186,34.83702163,34.83796971,34.83817906,34.83714126,34.83648987],
                  [31.27617381,31.27688681,31.27681264,31.27632574,31.27580774,31.27713744,31.27684853,31.27669002,31.27596147,31.27713864,31.27755017,31.27765904],
                  np.zeros(12)])
    
    with rasterio.open('ASTGTM2_N31E034_dem.tif') as src:
        for i in range(np.shape(x)[1]):
            x[:,i] = gps_map.geo2enu(x[1,i] , x[0,i] , [float(val[0]) for val in src.sample([(x[0,i],x[1,i])])][0]).T
    
    x_tag = np.vstack([x,np.ones(np.shape(x)[1])])
    
    uv_tag = np.array(np.dot(C,x_tag))
    y = np.array([uv_tag[0]/uv_tag[-1],uv_tag[1]/uv_tag[-1]])
    y_tag = np.vstack([y,np.ones(np.shape(y)[1])])
    
    P = dlt(x_tag,y_tag)
    print("||P-C|| = %.6g" %np.linalg.norm(P-C))
    #print('P   = ' + '\t['+']\n\t['.join([''.join([' {:10.4} '.format(item) for item in row]) for row in P.astype(float)])+']')
    
    H_inf = np.matrix(P[:,0:3])
    h_hat = np.array(P[:,3] , ndmin=2).T
    rc_enu = np.array(-H_inf.I*h_hat)
    #rc_zero = rc_enu
    rc_geo = gps_map.enu2geo(rc_enu[0],rc_enu[1],rc_enu[2])
    rc_zero = np.array([rc_geo[1] , rc_geo[0] , rc_geo[2]])
    
    differ = np.linalg.norm(rc_enu-gps_map.geo2enu(rc_real[1] , rc_real[0] , rc_real[2]))
    print("||rc-rc_real|| = %.6g" %differ)
    #print('rc  = ' + '\t{'+'}\n\t{'.join([''.join([' {:10.9} '.format(item) for item in row]) for row in rc_zero.astype(float)])+'}')
    #print('\trc_real  = ' + '\t{'+'}\n\t\t\t{'.join([''.join([' {:10.9} '.format(item) for item in row]) for row in rc_real.astype(float)])+'}')
    
    showy = np.hstack((rc_zero,rc_real))
    print('\t ______________ _____________')
    print('\t|      rc      |   rc_real   |')
    print('\t|--------------|-------------|')
    print('\t| '+'|\n\t| '.join([' | '.join([' {:10.9} '.format(item) for item in row]) for row in showy.astype(float)])+'|')
    print('\t|______________|_____________|')
    