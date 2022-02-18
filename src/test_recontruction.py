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
from setup_tools import parse_calib_file, parse_pickle
######## var #########

loader = PFMLoader(color=False, compress=False)
dataset_path = Path("datas/middlebury/chess1")
tune_params = Path("results/param.pkl")
calib_params = parse_calib_file(dataset_path/'calib.txt')
pprint(calib_params)
Q = np.array(
    [
        [1,         0,         0,              -calib_params["cam0"][0,2] ],
        [0,         1,         0,              -calib_params["cam0"][1,2] ],
        [0,         0,         0,               calib_params["cam0"][0,0] ],
        [0,         0,  1/calib_params["baseline"],                     0 ],
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




disparity = loader.load_pfm(dataset_path/'disp0.pfm')
# pfm1 = loader.load_pfm(dataset_path/'disp1.pfm')
img0 = cv2.imread((dataset_path/"im0.png").as_posix(), cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread((dataset_path/"im1.png").as_posix(), cv2.IMREAD_GRAYSCALE)

stereo_params = parse_pickle(tune_params)
pprint(stereo_params)



stereo = cv2.StereoBM_create()
stereo = cv2.StereoSGBM_create()
stereo.setNumDisparities(stereo_params["numDisparities"])
stereo.setBlockSize(stereo_params["blockSize"])
# stereo.setPreFilterType(param["preFilterType"])
# stereo.setPreFilterSize(param["preFilterSize"])
# stereo.setPreFilterCap(param["preFilterCap"])
# stereo.setTextureThreshold(param["textureThreshold"])
stereo.setUniquenessRatio(stereo_params["uniquenessRatio"])
stereo.setSpeckleRange(stereo_params["speckleRange"])
stereo.setSpeckleWindowSize(stereo_params["speckleWindowSize"])
stereo.setDisp12MaxDiff(stereo_params["disp12MaxDiff"])
stereo.setMinDisparity(stereo_params["minDisparity"])

# Compute the disparity image
disparity = stereo.compute(img0, img1)/16*2.
print(f"bound: [{disparity.min()}, {disparity.max()}]")
disparity[disparity <= 80] = np.nan
print(f"bound: [{disparity.min()}, {disparity.max()}]")
    
points_p = recontruction(disparity, Q)
point_cloud = pv.PolyData(points_p)
point_cloud.plot(eye_dome_lighting=True) #, render_points_as_spheres=True, point_size=1000.0)
