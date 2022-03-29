import numpy as np
import cv2
import configparser
from math import floor
import sys


def rectification():

    # Loading image to gray
    im_l = cv2.imread('{}'.format(sys.argv[1]))
    im_r = cv2.imread('{}'.format(sys.argv[2]))

    im_l_gray = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
    im_r_gray = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)


    
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
    cx_l = float(config.get('CAMERA_PARAMS_LEFT', 'cx'))
    cy_l = float(config.get('CAMERA_PARAMS_LEFT', 'cy'))    
    fx_l = float(config.get('CAMERA_PARAMS_LEFT', 'fx'))  
    fy_l = float(config.get('CAMERA_PARAMS_LEFT', 'fy'))  
    cameraMatrix1 = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]]) 

    
    # Matrice de calibration droite
    cx_r = float(config.get('CAMERA_PARAMS_RIGHT', 'cx'))
    cy_r = float(config.get('CAMERA_PARAMS_RIGHT', 'cy'))    
    fx_r = float(config.get('CAMERA_PARAMS_RIGHT', 'fx'))  
    fy_r = float(config.get('CAMERA_PARAMS_RIGHT', 'fy'))  
    cameraMatrix2 = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]]) 
    
    # Rectification epipolaire
    [R1, R2, P1, P2, Q, roi1, roi2] = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, resolution, R, T)
    
    # Im_left rectifiée
    mapl_x, mapl_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, im_l_gray.shape[::-1], cv2.CV_16SC2)
    rimg_l = cv2.remap(im_l, mapl_x, mapl_y,
                      interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
    cv2.imwrite('/results/rectified_im_l.png', rimg_l)
    
    # Im_r rectfiée
    mapr_x, mapr_y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, im_r_gray.shape[::-1], cv2.CV_16SC2)
    
    rimg_r = cv2.remap(im_r, mapr_x, mapr_y,
                      interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
    cv2.imwrite('/results/rectified_im_r.png', rimg_r)
    
    
rectification()
