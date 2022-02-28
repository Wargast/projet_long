import numpy as np
import cv2
import configparser

def rectification():

    
    # Loading image to gray
    im1 = cv2.imread('../datas/malaga/Images/img_CAMERA1_1261228749.918590_left.jpg')
    im2 = cv2.imread('../datas/malaga/Images/img_CAMERA1_1261228749.918590_right.jpg')
    
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    # Find the top 8 corners using the cv2.goodFeaturesToTrack()
    corners1 = cv2.goodFeaturesToTrack(im1_gray,8,0.01,10)
    corners1 = np.int0(corners1)
    
    corners2 = cv2.goodFeaturesToTrack(im2_gray,8,0.01,10)
    corners2 = np.int0(corners2)
    
    # Iterate over the corners and draw a circle at that location
    for i in corners1:
        x,y = i.ravel()
        cv2.circle(im1,(x,y),5,(0,0,255),-1)
        
    for i in corners2:
        x,y = i.ravel()
        cv2.circle(im2,(x,y),5,(0,0,255),-1)
        
    # Display the image
    cv2.imwrite('../results/im1_points.png',im1)
    
    cv2.imwrite('../results/im2_points.png',im2)
    
    # Matrice fondamentale
    F, mask = cv2.findFundamentalMat(corners1, corners2,cv2.FM_8POINT)
    rang_F = np.linalg.matrix_rank(F)
    
    # Rectification epipolaire
    [success,H1,H2] = cv2.stereoRectifyUncalibrated(corners1, corners2, F, (im1.shape[0],im1.shape[1]))
    
    
    ### Autre méthode sans F 
    # Loading fichier init
    config = configparser.ConfigParser()
    config.read('../datas/malaga/camera_params_raw_1024x768.ini')

    distCoeffs1 = np.array([float(i) for i in config.get('CAMERA_PARAMS_LEFT', 'dist')[1:-1].split(" ")])
    distCoeffs2 = np.array([float(i) for i in config.get('CAMERA_PARAMS_RIGHT', 'dist')[1:-1].split(" ")])
    
    # Loading R, T...
    R = np.array([[float(j) for j in i.split(" ")] for i in config.get('CAMERA_PARAMS_LEFT2RIGHT_POSE', 'rotation_matrix_only')[1:-2].split(" ;")])
    T = np.array([float(i) for i in config.get('CAMERA_PARAMS_LEFT2RIGHT_POSE','translation_only')[1:-1].split(" ")]) 
    resolution = np.array([int(i) for i in config.get('CAMERA_PARAMS_LEFT', 'resolution')[1:-1].split(" ")])
    
    # Matrice de calibration gauche
    cx_l = float(config.get('CAMERA_PARAMS_LEFT', 'cx')[1:-1])
    cy_l = float(config.get('CAMERA_PARAMS_LEFT', 'cy')[1:-1])    
    fx_l = float(config.get('CAMERA_PARAMS_LEFT', 'fx')[1:-1])  
    fy_l = float(config.get('CAMERA_PARAMS_LEFT', 'fy')[1:-1])  
    cameraMatrix1 = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]]) 
    
    # Matrice de calibration droite
    cx_r = float(config.get('CAMERA_PARAMS_RIGHT', 'cx')[1:-1])
    cy_r = float(config.get('CAMERA_PARAMS_RIGHT', 'cy')[1:-1])    
    fx_r = float(config.get('CAMERA_PARAMS_RIGHT', 'fx')[1:-1])  
    fy_r = float(config.get('CAMERA_PARAMS_RIGHT', 'fy')[1:-1])  
    cameraMatrix2 = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]]) 
    
    
          
    [R1, R2, P1, P2, Q, roi1, roi2] = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, resolution, R, T)
    
    
    return [R1, R2, P1, P2, Q, roi1, roi2]
    
H = rectification()
print(H)