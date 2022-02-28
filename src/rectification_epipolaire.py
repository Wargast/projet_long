import numpy as np
import cv2

def rectification():
    
    # Loading image to gray
    im1 = cv2.imread('/Users/alicepigneux/Desktop/Cours_3A/projet_long/datas/middlebury/artroom1/im0.png')
    im2 = cv2.imread('/Users/alicepigneux/Desktop/Cours_3A/projet_long/datas/middlebury/artroom1/im1.png')
    
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
    cv2.imwrite('/Users/alicepigneux/Desktop/Cours_3A/projet_long/results/im1_points.png',im1)
    
    cv2.imwrite('/Users/alicepigneux/Desktop/Cours_3A/projet_long/results/im2_points.png',im2)
    
rectification()