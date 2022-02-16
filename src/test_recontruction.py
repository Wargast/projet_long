from cmath import inf
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

############################

Q = np.array(
    [
        [1,  0,   0,    -cx],
        [0,  1,   0,    -cy],
        [0,  0,   0,     fx],
        [0,  0,  1/Tx,    0]
    ]
)


loader = PFMLoader(color=False, compress=False)
dataset_path = Path("datas/middlebury/ladder1")

pfm0 = loader.load_pfm(dataset_path/'disp0.pfm')
pfm1 = loader.load_pfm(dataset_path/'disp1.pfm')


points = []
tot_point = pfm0.shape[0] * pfm0.shape[1]
for u in range(pfm0.shape[0]):
    for v in range(pfm0.shape[1]):
        D = pfm0[u,v]
        p = np.dot(Q, np.array([[u,v,D,1]]).transpose())
        q = p/p[3]

        points.append(q[:3].transpose())
        print(f"{round((u*pfm0.shape[1]+v)/tot_point*100)}%", end="\r")

point_cloud = pv.PolyData( np.concatenate(points, axis=0) )
point_cloud.plot(eye_dome_lighting=True)
