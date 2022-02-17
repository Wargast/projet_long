from cmath import inf
import pickle
from pprint import pprint
import time
from unittest import result
from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import pyvista as pv
######## calib.txt ladder1 #########

fx = 1733.68
fy = 1733.68
cx = 819.72
cy = 957.55

doffs = 0
Tx = 221.13
width = 1920
height = 1080
ndisp = 100
vmin = 20
vmax = 70


Q = np.array(
    [
        [1,  0,   0,    -cx],
        [0,  1,   0,    -cy],
        [0,  0,   0,     fx],
        [0,  0,  1/Tx,    0]
    ]
)

############################

def recontruction(disparity, Q):
    xv, yv = np.meshgrid(range(disparity.shape[0]),range(disparity.shape[1]), indexing='ij')
    M = np.concatenate(
        [
            np.expand_dims(yv, axis=2),
            np.expand_dims(xv, axis=2),
            np.expand_dims(disparity, axis=2),
            np.expand_dims(np.ones(disparity.shape), axis=2),
        ], axis= 2)

    points = np.matmul(M,Q.transpose()).reshape(-1,4)
    points_p = np.multiply(points,(1/points[:, 3])[:,np.newaxis])
    points_p = points_p[:, :3]
    return points_p




loader = PFMLoader(color=False, compress=False)
dataset_path = Path("datas/middlebury/chess1")
tune_params = Path("results/param.pkl")

disparity = loader.load_pfm(dataset_path/'disp0.pfm')
# pfm1 = loader.load_pfm(dataset_path/'disp1.pfm')
img0 = cv2.imread((dataset_path/"im0.png").as_posix(), cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread((dataset_path/"im1.png").as_posix(), cv2.IMREAD_GRAYSCALE)

with open(tune_params, 'rb') as file:
      
    # Call load method to deserialze
    param = pickle.load(file)
  
    pprint(param)

numDisparities = param["numDisparities"]
blockSize = param["blockSize"]
preFilterType = param["preFilterType"]
preFilterSize = param["preFilterSize"]
preFilterCap = param["preFilterCap"]
textureThreshold = param["textureThreshold"]
uniquenessRatio = param["uniquenessRatio"]
speckleRange = param["speckleRange"]
speckleWindowSize = param["speckleWindowSize"]
disp12MaxDiff = param["disp12MaxDiff"]
minDisparity = param["minDisparity"]


stereo = cv2.StereoBM_create()
stereo.setNumDisparities(numDisparities)
stereo.setBlockSize(blockSize)
stereo.setPreFilterType(preFilterType)
stereo.setPreFilterSize(preFilterSize)
stereo.setPreFilterCap(preFilterCap)
stereo.setTextureThreshold(textureThreshold)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)
stereo.setMinDisparity(minDisparity)

# Compute the disparity image
disparity = stereo.compute(img0, img1)/16*2.
print(f"bound: [{disparity.min()}, {disparity.max()}]")
disparity[disparity <= 30] = np.nan
print(f"bound: [{disparity.min()}, {disparity.max()}]")
    
points_p = recontruction(disparity, Q)
point_cloud = pv.PolyData(points_p)
point_cloud.plot(eye_dome_lighting=True) #, render_points_as_spheres=True, point_size=1000.0)
