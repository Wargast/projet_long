from pathlib import Path
import cv2
import numpy as np
from local_matching import Local_matching
from pypfm import PFMLoader
from matplotlib import pyplot as plt
loader = PFMLoader(color=False, compress=False)

########## Param ##########
numDisparities = 16*20
blockSize=21
dataset_path = Path("datas/middlebury/chess1")
###########################


i_left = cv2.imread("datas/middlebury/curule1/im0.png", cv2.IMREAD_GRAYSCALE)
i_right = cv2.imread("./datas/middlebury/curule1/im1.png", cv2.IMREAD_GRAYSCALE)

# stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
# stereo = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
# disparity_map = stereo.compute(i_left, i_right)/16

stereo = Local_matching(max_disparity=128, block_size=21, method="census", seuil_symmetrie=0)
disparity_map = stereo.compute(i_left, i_right)
disparity_map[disparity_map==np.inf] = 0

disparity_map_GS = loader.load_pfm("./datas/middlebury/curule1/disp0.pfm")

plt.figure(dataset_path.as_posix())#, figsize=(160,90))
plt.subplot(1,2,1)
plt.imshow(disparity_map, 'gray')
plt.title(f"disparity map with numDisparities={numDisparities}, blockSize={blockSize}")
plt.subplot(1,2,2)
plt.imshow(disparity_map_GS, 'gray')
plt.title("Goal Stantard disparity map")
plt.show()

exit()
