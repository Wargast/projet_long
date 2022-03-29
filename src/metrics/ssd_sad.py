import numpy as np
from tqdm import tqdm
import cv2
import time
import ssd_sad_c
import matplotlib.pyplot as plt


def ssd_sad_numpy(
    im_left: np.ndarray, im_right: np.ndarray, block_size: int, max_disparity: int
) -> np.ndarray:

    (h1, l1) = im_right.shape
    (h2, l2) = im_left.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1
    l2_cropped = l2 - block_size + 1

    blocks_l = np.lib.stride_tricks.sliding_window_view(im_left, (block_size, block_size))
    blocks_l = blocks_l.reshape(h1_cropped, l1_cropped, block_size ** 2)

    blocks_r = np.lib.stride_tricks.sliding_window_view(im_right, (block_size, block_size))
    blocks_r = blocks_r.reshape(h1_cropped, l2_cropped, block_size ** 2)

    error_map_l = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )

    error_map_r = np.full(
        (h1_cropped, l2_cropped, max_disparity + 1),
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )

    for d in tqdm(range(max_disparity + 1)):
        right = blocks_r[:, 0 : min(l2_cropped - d, l1_cropped), :]
        left = blocks_l[:, d:, :]

        costs = np.sum(np.square((right.astype(np.int32) - left.astype(np.int32))), axis=2)

        error_map_r[:, 0 : min(l2_cropped - d, l1_cropped), d] = costs
        error_map_l[:, d:, d] = costs

    return error_map_l, error_map_r
    


if __name__ == "__main__":
    im_left = cv2.imread("./datas/middlebury/curule1/im0.png", cv2.IMREAD_GRAYSCALE)
    im_right = cv2.imread("./datas/middlebury/curule1/im1.png", cv2.IMREAD_GRAYSCALE)

    # im_left = np.random.randint(0, 255, size=(4,4), dtype=np.uint8)
    # im_right = np.random.randint(0, 255, size=(4,4), dtype=np.uint8)

    block_size = 9
    max_disparity = 150

    # error_map_numpy_l, error_map_numpy_r = ssd_sad_numpy(
    #     im_left, im_right, block_size, max_disparity
    # )

    a = time.perf_counter()
    error_map_cython_l, error_map_cython_r = ssd_sad_c.ssd_sad(
        im_left, im_right, block_size, max_disparity
    )
    b = time.perf_counter()

    print(f"temps de calcul ssd cython : {b - a} s")

    # print(np.array_equal(error_map_numpy_r, error_map_cython_r))
    # print(np.array_equal(error_map_numpy_l, error_map_cython_l))

    disparity = ssd_sad_c.disparity_from_error_map(error_map_cython_l, error_map_cython_r, block_size, seuil_symmetrie=0)

    disparity[disparity==np.inf] = 0

    plt.imshow(disparity, 'gray')
    plt.show()

