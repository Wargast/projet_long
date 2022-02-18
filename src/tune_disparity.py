from cmath import inf
import time
from unittest import result
# from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import pyvista as pv
import pickle
from pprint import pprint


def nothing(a):
    pass

def save_param(*args):
    params = {
        'numDisparities': numDisparities,
        'blockSize': blockSize,
        'preFilterType': preFilterType,
        'preFilterSize': preFilterSize,
        'preFilterCap': preFilterCap,
        'textureThreshold': textureThreshold,
        'uniquenessRatio': uniquenessRatio,
        'speckleRange': speckleRange,
        'speckleWindowSize': speckleWindowSize,
        'disp12MaxDiff': disp12MaxDiff,
        'minDisparity': minDisparity,
    }
    with open(tune_params, 'wb') as file:
        pickle.dump(params, file)
        pprint(params)


cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)


# Creating an object of StereoBM algorithm
# stereo = cv2.StereoBM_create()
stereo = cv2.StereoSGBM_create()

# loader = PFMLoader(color=False, compress=False)
dataset_path = Path("datas/middlebury/chess1")
tune_params = Path("results/param.pkl")

# disparity = loader.load_pfm(dataset_path/'disp0.pfm')
# pfm1 = loader.load_pfm(dataset_path/'disp1.pfm')
imgL_gray = cv2.imread((dataset_path/"im0.png").as_posix(), cv2.IMREAD_GRAYSCALE)
imgR_gray = cv2.imread((dataset_path/"im1.png").as_posix(), cv2.IMREAD_GRAYSCALE)

with open(tune_params, 'rb') as file:
      
    # Call load method to deserialze
    param = pickle.load(file)
  
    pprint(param)

numDisparities = param["numDisparities"]/16
blockSize = (param["blockSize"]- 5)/2
preFilterType = param["preFilterType"]
preFilterSize = (param["preFilterSize"]- 5)/2
preFilterCap = param["preFilterCap"]
textureThreshold = param["textureThreshold"]
uniquenessRatio = param["uniquenessRatio"]
speckleRange = param["speckleRange"]
speckleWindowSize = param["speckleWindowSize"]/2
disp12MaxDiff = param["disp12MaxDiff"]
minDisparity = param["minDisparity"]

cv2.createTrackbar('numDisparities','disp',int(numDisparities),50,nothing)
cv2.createTrackbar('blockSize','disp',int(blockSize),50,nothing)
cv2.createTrackbar('preFilterType','disp',int(preFilterType),1,nothing)
cv2.createTrackbar('preFilterSize','disp',int(preFilterSize),25,nothing)
cv2.createTrackbar('preFilterCap','disp',int(preFilterCap),62,nothing)
cv2.createTrackbar('textureThreshold','disp',int(textureThreshold),100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',int(uniquenessRatio),100,nothing)
cv2.createTrackbar('speckleRange','disp',int(speckleRange),100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',int(speckleWindowSize),25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',int(disp12MaxDiff),25,nothing)
cv2.createTrackbar('minDisparity','disp',int(minDisparity),25,nothing)
cv2.createButton("save param", save_param, None,cv2.QT_PUSH_BUTTON,1)

while True:

    # Applying stereo image rectification on the left image
    Left_nice= imgL_gray
    
    # Applying stereo image rectification on the right image
    Right_nice= imgR_gray

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
    
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    # stereo.setPreFilterType(preFilterType)
    # stereo.setPreFilterSize(preFilterSize)
    # stereo.setPreFilterCap(preFilterCap)
    # stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity using the StereoBM algorithm

    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Converting to float32 
    disparity = disparity.astype(np.float32)

    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities

    # Displaying the disparity map
    cv2.imshow("disp",disparity)
    print("OK!")
    

    # Close window using esc key
    key = cv2.waitKey()
    if key == 27:
        break