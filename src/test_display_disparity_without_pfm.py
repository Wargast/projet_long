from pathlib import Path
import cv2
import numpy as np
from pfm import read_pfm
from matplotlib import pyplot as plt


########## Param ##########
numDisparities = 16*20
blockSize=21
dataset_path = Path("datas/middlebury/chess1")

###########################


i_left = cv2.imread("datas/middlebury/chess1/im0.png", cv2.IMREAD_GRAYSCALE)
i_right = cv2.imread("./datas/middlebury/chess1/im1.png", cv2.IMREAD_GRAYSCALE)

# stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
stereo = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)

# Compute the disparity image
disparity_map = stereo.compute(i_left, i_right)/16
disparity_map_GS = read_pfm("./datas/middlebury/chess1/disp0.pfm")

disparity_map_GS = cv2.flip(disparity_map_GS, 0)
plt.figure(dataset_path.as_posix())#, figsize=(160,90))
plt.subplot(1,2,1)
plt.imshow(disparity_map, 'gray')
plt.title(f"disparity map with numDisparities={numDisparities}, blockSize={blockSize}")
plt.subplot(1,2,2)
plt.imshow(disparity_map_GS, 'gray')
plt.title("Goal Stantard disparity map")
plt.show()

exit()


im = disparity_map / np.max(disparity_map)

im = (im * 255).astype(np.uint8)

# im = cv2.resize(im, (540, 960))

cv2.imshow('test', im)

cv2.waitKey(0)
