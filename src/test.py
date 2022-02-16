import cv2
import numpy as np

disparity_map = np.load('disp_census_artroom1.npy')

disparity_map[disparity_map == np.inf] = 0

im = disparity_map / np.max(disparity_map)

im = (im * 255).astype(np.uint8)

# im = cv2.resize(im, (540, 960))

cv2.imshow('test', im)

cv2.waitKey(0)
