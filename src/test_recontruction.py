from cmath import inf
import pickle
from pprint import pprint
import time
from unittest import result
from local_matching import Local_matching
from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import pyvista as pv
from setup_tools import (parse_calib_file, parse_pickle,
                        stereoBM_from_file, stereoSGBM_from_file)

######## var #########

loader = PFMLoader(color=False, compress=False)
dataset_path = Path("datas/middlebury/curule1")
tuned_params = Path("results/param.pkl")
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
disparity = np.squeeze(disparity)
disparity = np.flipud(disparity)
# pfm1 = loader.load_pfm(dataset_path/'disp1.pfm')
img0 = cv2.imread((dataset_path/"im0.png").as_posix(), cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread((dataset_path/"im1.png").as_posix(), cv2.IMREAD_GRAYSCALE)


# stereo = stereoSGBM_from_file(tuned_params)
# stereo = stereoBM_from_file(tuned_params)
# disparity = stereo.compute(img0, img1)/16

stereo = Local_matching(max_disparity=128, cost_threshold=1)
disparity = stereo.compute(img0, img1)

print(f"bound: [{disparity.min()}, {disparity.max()}], nb points:{np.sum(~np.isnan(disparity))}")
disparity[disparity <= 12] = np.nan
disparity[disparity >= 300] = np.nan
print(f"bound: [{disparity.min()}, {disparity.max()}], nb points:{np.sum(~np.isnan(disparity))}")
    
points_p = recontruction(disparity, Q)
point_cloud = pv.PolyData(points_p)
point_cloud.plot(eye_dome_lighting=True) #, render_points_as_spheres=True, point_size=1000.0)
